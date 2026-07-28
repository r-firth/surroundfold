#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 PATH_TO_SURROUNDFOLD [WORK_DIRECTORY]" >&2
  exit 2
fi

surroundfold="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
if [[ ! -x "$surroundfold" ]]; then
  echo "surroundfold executable is missing or not executable: $surroundfold" >&2
  exit 2
fi

remove_workspace=false
if [[ $# -eq 2 ]]; then
  workspace="$2"
  mkdir -p "$workspace"
else
  workspace="$(mktemp -d)"
  remove_workspace=true
fi
cleanup() {
  if [[ "$remove_workspace" == true ]]; then
    rm -rf -- "$workspace"
  fi
}
trap cleanup EXIT

ffmpeg_path="$(command -v ffmpeg)"
ffprobe_path="$(command -v ffprobe)"
source_path="$workspace/synthetic-source.mkv"
output_path="$workspace/synthetic-output.mkv"

"$ffmpeg_path" -y -hide_banner -loglevel error \
  -f lavfi -i "testsrc2=size=96x54:rate=5:duration=0.25" \
  -f lavfi -i "sine=frequency=440:sample_rate=48000:duration=0.25" \
  -map 0:v -map 1:a \
  -c:v ffv1 \
  -filter:a "pan=5.1|c0=c0|c1=c0|c2=c0|c3=c0|c4=c0|c5=c0" \
  -c:a eac3 -b:a 640k \
  -metadata title="Self-contained package smoke" \
  -metadata:s:a:0 language=eng \
  "$source_path"

"$surroundfold" "$source_path" \
  --output "$output_path" \
  --ffmpeg "$ffmpeg_path" \
  --ffprobe "$ffprobe_path" \
  --progress quiet

audio_codecs="$("$ffprobe_path" -v error -select_streams a \
  -show_entries stream=codec_name -of csv=p=0 "$output_path" | tr -d '\r')"
if ! grep -Fxq "eac3" <<<"$audio_codecs" ||
    ! grep -Fxq "pcm_s16le" <<<"$audio_codecs"; then
  echo "published smoke output did not preserve E-AC-3 and append PCM" >&2
  exit 1
fi

echo "release smoke passed: $output_path"
