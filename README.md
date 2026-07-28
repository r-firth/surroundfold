# SurroundFold

A offline binaural renderer written in Rust. It renders ordinary
channel audio, TrueHD beds and moving objects, and Dolby Digital Plus/Atmos JOC
objects through a bundled or custom HRIR, then appends the stereo result to a
new audio track in a .mkv file.

## Install

Install Rust 1.88 or newer and FFmpeg:

```bash
brew install ffmpeg
rustup toolchain install 1.88.0
cargo build --release --locked
```

The executable is `target/release/surroundfold`. A universal macOS package can be
built on a Mac with:

```bash
scripts/publish-all.sh
```

## Use

```bash
target/release/surroundfold "Movie.mkv" \
  --output "Movie.binaural.mkv"
```

The bundled 48 kHz, 7.1 HRIR is used by default. Pass
`--hrir "My HRIR.wav"` to override it. The default provides horizontal
surround virtualization; use a profile containing height positions for
individualized elevation cues.

If `--output` is omitted, the result is
`<input-name>.surroundfold.mkv` beside the input. Existing output is never
replaced unless `--overwrite` is supplied, and even then the old file remains
until the replacement has passed post-mux verification.

List the input’s audio tracks:

```bash
target/release/surroundfold "Movie.mkv" --list-tracks
```

Automatic selection prefers object-capable TrueHD, then DD+/Atmos JOC, then
ordinary channel codecs. `--track INDEX` selects an explicit zero-based audio
track. `--language CODE` makes the language a requirement rather than a
fallback preference.

Custom HRIRs use a stereo WAV whose two channels contain concatenated left/right
impulse responses. Layouts with 1 through 16 virtual positions are accepted.
The custom HRIR and object stream must have the same sample rate. Ordinary
channel audio is resampled by FFmpeg to the HRIR rate.

## Render controls

| Option | Default | Purpose |
| --- | --- | --- |
| `--hrir PATH` | bundled profile | Override the embedded stereo HRIR |
| `--gain-db NUMBER` | `0` | Gain before the output limiter |
| `--surround-swap on\|off` | `off` | Exchange side and rear surround routes |
| `--speaker-virtualizer on\|off` | `off` | Direct virtual-loudspeaker rendering for ground beds |
| `--mute-bed on\|off` | `off` | Mute reference bed sources |
| `--mute-ground on\|off` | `off` | Mute ground-plane sources |
| `--room-correction PATH` | off | Apply a stereo room-correction FIR after binaural rendering |
| `--matrix on\|off` | `off` | Expand channel audio before height generation |
| `--upconvert on\|off` | `off` | Generate height content from channel audio |
| `--effect 0..100` | `75` | Height-generation strength |
| `--smoothness 0..100` | `80` | Generated-height movement smoothing |
| `--mlp-presentation 0..3` | `3` | TrueHD presentation |
| `--unsafe-parsing` | off | Relax supported bitstream validation |

Matrix expansion and generated height are channel-audio features. They are
rejected for native object tracks rather than silently ignored.

Run `surroundfold --help` for path, progress, temporary-file, overwrite, and
advanced mux options.

## What the renderer preserves

The input is opened read-only. The output maps every source stream, metadata,
chapter, attachment, and disposition; every original stream is copied without
transcoding. Only the appended binaural track is encoded, as stereo
`pcm_s16le`.

Before publishing the output, the program probes the partial file and verifies:

- the original stream count, order, codecs, tags, and dispositions;
- global metadata and chapters;
- the appended codec, channel count, sample rate, title, and language;
- selected-track and binaural start-time alignment;
- exact rendered duration, including the HRIR and room-FIR tail.

Output publication is an atomic rename. Cancellation, decode failure, mux
failure, and verification failure leave no partial file at the requested path.

## Architecture

The main paths are deliberately separate at the codec boundary and share one
renderer after producing PCM plus spatial updates:

```text
TrueHD ── embedded Rust decoder ─────┐
                                    ├─ object panner ─┐
E-AC-3/JOC ─ FFmpeg core + Rust JOC ┘                │
                                                     ├─ HRIR convolution
channel codecs ───── FFmpeg PCM ─ channel processing ┘
                                                       │
                               limiter + TPDF dither ──┴─ preservation mux
```

The JOC path implements OAMD and JOC EMDF extraction, dense and sparse Huffman
matrix modes, differential dequantization, temporal interpolation, the
normative complex QMF analysis/synthesis pair, LFE bypass alignment, and
sample-accurate metadata scheduling. Its fixed 577-sample analysis/synthesis
latency is removed so the appended track remains aligned with the source.

## Test and quality gates

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo test --workspace --all-targets --no-default-features
```

Generated integration tests exercise channel rendering, TrueHD decoding when
the local FFmpeg supplies its experimental encoder, preservation muxing,
advanced controls, and both feature configurations.

Real codec samples stay outside the repository. Optional regression fixtures
are enabled with:

```bash
SURROUNDFOLD_ATMOS_SAMPLE=/path/to/sample.thd \
  cargo test --test atmos_fixture -- --nocapture

SURROUNDFOLD_EAC3_JOC_SAMPLE=/path/to/sample.ec3 \
  cargo test --test eac3_fixture -- --nocapture
```

The E-AC-3 fixture test parses every OAMD/JOC payload, checks finite
reconstruction matrices and object counts, performs the full QMF/object render,
and verifies the preserved Matroska output is non-silent.

CI runs formatting, strict Clippy, both feature configurations, generated
end-to-end renders, release builds, smoke tests, dependency-source policy,
license policy, and RustSec advisory checks on Apple Silicon and Intel macOS.

## Current boundaries

- The embedded `truehd` 0.4.0 decoder is experimental upstream. Strict parsing
  is the default, errors are fatal, and decoder panics are contained and
  reported, but broad real-title coverage is still important.
- TrueHD intermediate-spatial-format objects are not yet rendered.
- DD+/Atmos currently requires one OAMD and one JOC payload per complete
  program frame, and supports the five-channel, seven-channel, and height
  downmix configurations defined by ETSI TS 103 420.
- SOFA HRIR import, object-domain resampling, live playback, and hardware
  bitstream output are not implemented.
