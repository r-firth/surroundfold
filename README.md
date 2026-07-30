# SurroundFold

An offline binaural renderer written in Rust. It renders ordinary
channel audio, TrueHD beds and moving objects, and Dolby Digital Plus/Atmos JOC
objects through a bundled or custom HRIR, then appends the stereo result to a
new non-default audio track in the source .mkv file.

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
target/release/surroundfold "Movie.mkv"
```

The default operation is in place: SurroundFold builds and verifies a complete
replacement beside the source, then atomically swaps it over the original.
Pass `--output "Movie.binaural.mkv"` to keep the source and write a separate
file instead.

The bundled 48 kHz, 7.1 HRIR is used by default and is resampled once when the
selected track uses another rate. Pass
`--hrir "My HRIR.sofa"` or `--hrir "My HRIR.wav"` to override it. The default
provides horizontal surround virtualization; a measured SOFA profile provides
individualized elevation cues.

An existing explicit `--output` is not replaced unless `--overwrite` is
supplied. In every mode, the old file remains available until the replacement
has passed post-mux verification.

List the input’s audio tracks:

```bash
target/release/surroundfold "Movie.mkv" --list-tracks
```

Automatic selection prefers object-capable TrueHD, then DD+/Atmos JOC, then
ordinary channel codecs. `--track INDEX` selects an explicit zero-based audio
track. `--language CODE` makes the language a requirement rather than a
fallback preference.

SOFA HRIRs are interpolated into a 66-direction full-sphere array: 18 exact
reference-layout routes plus 48 supplemental horizontal, upper, and lower
routes for finer object motion. The normal renderer replaces profile-wide
colour with a clean analytic ITD/ILD body, then retains up to ±2.5 dB of
smoothed, zero-mean measured direction shape between 2.5 and 18 kHz. That
common-ear direction shape adds rear and height identity without importing the
profile's dark or reverberant character. Concatenated stereo WAV profiles
remain supported with 1 through 16 virtual positions and receive the same
one-time high-quality impulse resampling.

The single appended track receives a fixed common-left/right finishing curve:
+0.8 dB at 60 Hz, -0.8 dB around 240 Hz, and a +0.5 dB high shelf from 8 kHz.
The renderer applies it before linked true-peak control and final 24-bit
quantization, so the delivery encode cannot introduce another rounding pass.
It adds no compression, loudness normalization, or channel-dependent
processing.

## Render controls

| Option | Default | Purpose |
| --- | --- | --- |
| `--hrir PATH` | bundled profile | Override the embedded HRIR with SOFA or concatenated WAV |
| `--gain-db NUMBER` | `-5.5` | Gain before the output limiter |
| `--surround-swap on\|off` | `off` | Exchange side and rear surround routes |
| `--speaker-virtualizer on\|off` | `off` | Direct stereo fold-down for ground beds, bypassing their HRIRs |
| `--mute-bed on\|off` | `off` | Mute reference bed sources |
| `--mute-ground on\|off` | `off` | Mute ground-plane sources |
| `--room-correction PATH` | off | Apply a stereo room-correction FIR after binaural rendering |
| `--output-codec flac\|aac` | `flac` | Lossless 24-bit delivery, or fast 320 kb/s AAC-LC compatibility output |
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

The input is opened read-only. The replacement maps every source stream,
metadata, chapter, attachment, and disposition; every original stream is
copied without transcoding. The appended binaural track defaults to source-rate,
24-bit stereo FLAC at compression level 0. This is sample-exact to the finished
renderer output; the low compression setting minimizes encoding time and
changes only file size. `--output-codec aac` instead selects 320 kb/s AAC-LC
with FFmpeg's fast search mode for players that need a compatibility track.
The appended track is always non-default. Running SurroundFold again replaces
its earlier output instead of accumulating duplicate tracks.

The preservation mux is streamed in two stages. The approved finishing filter
has already run inside the Rust renderer; the lightweight FLAC or optional AAC
encode runs in the first mux stage while video and original audio are copied
and interleaved. Sparse subtitles, data, and attachments are copied back in
the second stage. This prevents PGS subtitle gaps from leaving appended audio
packets hundreds of megabytes apart, which can starve real-time hardware
demuxers even when an offline decoder accepts the file.

Before publishing the output, the program probes the partial file and verifies:

- the original stream count, order, codecs, tags, and dispositions;
- global metadata and chapters;
- the appended codec, channel count, sample rate, title, and language;
- selected-track and binaural start-time alignment;
- rendered duration, including the HRIR, early-reflection, room-FIR, codec
  priming where applicable, and final padded-frame tails.

Output publication is an atomic same-filesystem rename which preserves the
original file mode. Cancellation, decode failure, mux failure, and verification
failure leave the original untouched.

## Architecture

The main paths are deliberately separate at the codec boundary and share one
renderer after producing PCM plus spatial updates:

```text
TrueHD ─ FFmpeg byte stream ─ Rust decoder ─┐
                                           ├─ object panner ─┐
E-AC-3/JOC ─ FFmpeg core + Rust JOC ┘                │
                                                     ├─ HRIR convolution
channel codecs ───── FFmpeg PCM ─ channel processing ┘
                                                       │
 common finishing EQ ─ linked true-peak limiter ─ 24-bit TPDF dither
                                                       │
                              FLAC (default) or AAC compatibility ─ preservation mux
```

The JOC path implements OAMD and JOC EMDF extraction, dense and sparse Huffman
matrix modes, differential dequantization, temporal interpolation, the
normative complex QMF analysis/synthesis pair, LFE bypass alignment, and
sample-accurate metadata scheduling. Its fixed 577-sample analysis/synthesis
latency is removed so the appended track remains aligned with the source.
TrueHD and DD+/Atmos beds and objects share calibrated binaural LFE routing,
parametric ITD/ILD panning with bounded measured direction shapes,
metadata-space movement interpolation, and linked 4×-oversampled true-peak
control. Authored OAMD distance is retained separately from direction and
controls a sparse directional early field. Four arrivals
between roughly 7 and 34 milliseconds add externalization and distance cues
without attenuating or delaying the authored direct signal; no late reverb is
added.

## Test and quality gates

```bash
cargo fmt --all -- --check
cargo clippy -p surroundfold --all-targets --no-deps -- -D warnings
cargo test --workspace --all-targets
cargo test --workspace --all-targets --no-default-features
```

Generated integration tests exercise channel rendering, TrueHD decoding when
the local FFmpeg supplies its experimental encoder, 48 and 96 kHz lossless PCM
agreement between both decoders, preservation muxing, advanced controls, and
both feature configurations. The mux regressions prove that default FLAC
decodes to the finished 24-bit samples byte-for-byte and that explicit AAC uses
the approved compatibility settings. A synthetic high-bitrate fixture with
sixteen sparse PGS streams also checks the physical spacing of appended FLAC
packets, not merely whether FFmpeg can decode them afterward.

The small Apache-2.0 TrueHD decoder dependency is pinned in
`rust/vendor/truehd`; its focused regression tests cover local metadata
correctness fixes that are not yet available in its published crate.

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
- DD+/Atmos accepts the one OAMD and one JOC payload per codec frame required
  by ETSI TS 103 420, including all five-channel, seven-channel, and height
  downmix configurations defined there.
- The bundled 7.1 profile has no measured overhead responses. Supply a measured
  SOFA profile for genuine individualized elevation cues.
- SOFA object rendering uses an artifact-free 66-direction virtual array rather
  than swapping a time-varying convolution filter at every metadata update.
- Live playback and hardware bitstream output are outside this offline
  stereo-renderer's scope.
