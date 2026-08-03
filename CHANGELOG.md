# Changelog

All notable changes to SurroundFold are recorded here. Version headings and
conventional-commit entries are maintained by Commitizen.

## v0.1.1 (2026-08-03)

### Fix

- prevent sparse subtitles starving appended audio

## v0.1.0 (2026-08-02)

### Added

- Offline, in-place rendering that preserves every original Matroska stream and
  appends a verified non-default binaural stereo track.
- Native Rust TrueHD/Atmos decoding and Dolby Digital Plus/Atmos JOC object
  reconstruction, alongside ordinary channel-audio rendering.
- A bundled default HRIR plus custom SOFA and concatenated-WAV profile support.
- Continuous point-object rendering with analytic ITD/ILD cues and retained
  measured directional shape.
- Authored-distance rendering with bounded early reflections and an optional
  image-source model.
- Subtle linked stereo finishing, 24-bit lossless FLAC delivery, and optional
  320 kb/s AAC-LC compatibility output.
- Atomic replacement, cancellation safety, preservation verification, and
  generated end-to-end codec and mux tests.
- Five-second render and mux progress updates with percentage, real-time speed,
  and ETA estimates.
- Native Apple Silicon and Intel builds packaged as one universal macOS
  executable, plus a portable statically linked Linux x86-64 build.

### Fixed

- AAC priming validation now handles the timestamp conventions reported by both
  FFmpeg 8.0 and 8.1 without accepting genuine synchronization offsets.

### Changed

- Continuous object rendering and source-relative image reflections are now
  the defaults; both baseline rendering modes remain available explicitly.
- Continuous object rendering now uses channel-indexed state, allocation-free
  filter targets, cached stationary HRTF/delay ramps, and finite-tail silence
  detection. Render time fell by roughly 52–53% across both Atmos test clips
  without changing the approved DSP path.
- Stereo continuous filtering, fractional delay, true-peak history, VBAP
  scratch use, reflection taps, and FFT input clearing now share or reuse their
  fixed storage. A further profiling pass reduced render time by roughly 8–9%
  on the Atmos fixtures with bit-identical finished PCM.
- Preservation now uses one verified FFmpeg pass instead of remuxing through a
  second process. The sparse-PGS packet-spacing invariant remains covered while
  local encode-and-mux time falls by roughly 43–45%.
