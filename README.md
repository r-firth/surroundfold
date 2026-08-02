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

On a Linux media server whose CPU supports the x86-64-v3 feature level, an
explicitly tuned build can be substantially faster than the portable x86-64
binary:

```bash
CARGO_ENCODED_RUSTFLAGS='-Ctarget-cpu=x86-64-v3' \
  cargo zigbuild --release --locked --target x86_64-unknown-linux-gnu
```

Tagged releases provide a universal Apple Silicon and Intel macOS package and
a statically linked, portable Linux x86-64 package on the
[GitHub releases page](https://github.com/r-firth/surroundfold/releases).
Both packages require `ffmpeg` and `ffprobe` on `PATH` at runtime.

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

For object-capable tracks, `--object-renderer continuous` replaces
virtual-speaker blending for point objects with one time-aligned binaural
filter per object. Analytic ILD and the common-ear directional spectral shape
are synthesized from separate targets into a short minimum-phase body on a
5-by-10-degree spherical grid, while ITD is applied as a separate smoothly
varying 24-tap bandlimited fractional delay. The delay follows rigid-sphere
Woodworth path geometry instead of scaling the ear maximum as a sine, avoiding
exaggerated ITD at intermediate lateral angles. Both interaural cues use the
physical 3-D interaural-axis projection, so they weaken continuously toward
the elevation poles. Unconstrained point objects feed their authored 3-D
direction directly to this path, preserving height even when the selected
virtual-speaker profile has only ground-plane routes. Explicit snap and zone
constraints retain their metadata-defined route bearing. This prevents
intermediate positions from mixing several already-delayed HRIRs, flattening
height through a coplanar route array, or losing upper-spectrum energy to
short delay interpolation. Speaker-anchored and deliberately extended objects
retain the established route renderer.

`--distance-renderer image-source` independently replaces the fixed early
field. It uses constant-power direct/early scaling and six first-order image
reflections whose path-excess delay, arrival direction, surface loss, and
high-frequency absorption follow each source. Reflection timing uses the same
bandlimited fractional delay as moving ITD. The model has no late-reverb stage
and keeps every reflection arrival inside 80 milliseconds; filter decay is
scaled with sample rate so high-rate renders are not truncated above the
24-bit floor. Continuous object rendering and image-source distance rendering
are the defaults; the baseline modes remain available for direct comparison.

The single appended track receives a fixed common-left/right finishing curve:
+0.8 dB from a broad bell centred at 55 Hz, -0.8 dB around 240 Hz, and a
+0.5 dB high shelf from 8 kHz.
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
| `--object-renderer baseline\|continuous` | `continuous` | Virtual-speaker routes or continuous point-object filters |
| `--distance-renderer baseline\|image-source` | `image-source` | Fixed early field or source-relative first-order reflections |
| `--mute-bed on\|off` | `off` | Mute reference bed sources |
| `--mute-ground on\|off` | `off` | Mute ground-plane sources |
| `--room-correction PATH` | off | Apply a stereo room-correction FIR after binaural rendering |
| `--output-codec flac\|aac` | `flac` | Lossless 24-bit delivery, or fast 320 kb/s AAC-LC compatibility output |
| `--track-title TITLE` | `SurroundFold binaural` | Override the appended track's Matroska title |
| `--keep-existing-surroundfold` | off | Preserve tracks appended by earlier SurroundFold runs |
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

Text progress reports render and mux percentage, real-time speed, and ETA every
five seconds. `--progress quiet` suppresses those updates, while
`--progress json` keeps standard output machine-readable and includes final
phase timings.

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

The preservation mux runs in one FFmpeg pass. The approved finishing filter has
already run inside the Rust renderer; only the lightweight FLAC or optional AAC
delivery track is encoded, while every original packet is copied. Video,
original audio, and the appended track are mapped before sparse subtitles,
data, and attachments so FFmpeg's bounded interleaver writes the new audio at
hardware-friendly intervals. The generated 16-stream PGS fixture verifies both
the first packet position and the maximum physical gap between appended audio
packets.

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
control. The baseline distance renderer retains authored OAMD distance
separately from direction and feeds four fixed arrivals between roughly 7 and
34 milliseconds. The default image-source renderer instead uses the authored
distance to balance direct and early energy while moving six first-order
arrivals with the object. Neither path adds late reverb.

## Test and quality gates

```bash
cargo fmt --all -- --check
cargo clippy -p surroundfold --all-targets --no-deps -- -D warnings
cargo test --workspace --all-targets
cargo test --workspace --all-targets --no-default-features
```

Repeat the four-way renderer wall-clock regression on a representative object
track with:

```bash
scripts/benchmark-renderers.sh /path/to/object-audio.mkv
```

Generated integration tests exercise channel rendering, TrueHD decoding when
the local FFmpeg supplies its experimental encoder, 48 and 96 kHz lossless PCM
agreement between both decoders, preservation muxing, advanced controls, and
both feature configurations. The mux regressions prove that default FLAC
decodes to the finished 24-bit samples byte-for-byte and that explicit AAC uses
the approved compatibility settings. A synthetic high-bitrate fixture with
sixteen sparse PGS streams also checks the physical spacing of appended FLAC
packets, not merely whether FFmpeg can decode them afterward.

The TrueHD decoder dependency is pinned in `rust/vendor/truehd`; its focused
regression tests cover local metadata correctness fixes that are not yet
available in its published crate.

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
end-to-end renders, release builds, and smoke tests on Apple Silicon, Intel
macOS, and portable static Linux x86-64.

## Releasing

Use conventional commit messages (`feat:`, `fix:`, and so on), then run the
`release` workflow from the Actions page when the next release is ready.
Commitizen derives the semantic version from commits since the previous tag;
there are no version files or tags to edit manually.

The first run publishes the existing `0.1.0`. Later runs atomically update the
Cargo workspace version, `Cargo.lock`, and `CHANGELOG.md`, then commit and tag
that bump. The workflow reruns the full quality gate, builds and smoke-tests
both macOS architectures plus static Linux x86-64, creates the ad-hoc-signed
universal macOS executable, publishes both versioned archives and SHA-256
checksums, and uses the generated changelog section as the GitHub release
notes.

## Current boundaries

- The embedded `truehd` 0.4.0 decoder is experimental upstream. Strict parsing
  is the default, errors are fatal, and decoder panics are contained and
  reported, but broad real-title coverage is still important.
- DD+/Atmos accepts the one OAMD and one JOC payload per codec frame required
  by ETSI TS 103 420, including all five-channel, seven-channel, and height
  downmix configurations defined there.
- The bundled 7.1 profile has no measured overhead responses. Supply a measured
  SOFA profile for genuine individualized elevation cues.
- The default continuous renderer bypasses the profile's 66-direction virtual
  array for point objects and preserves their authored elevation independently
  of the array. Speaker-anchored and deliberately extended objects still use
  the array, and `--object-renderer baseline` restores it for all objects.
- Live playback and hardware bitstream output are outside this offline
  stereo-renderer's scope.
