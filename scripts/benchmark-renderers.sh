#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 INPUT_MKV [SURROUNDFOLD_BINARY]" >&2
  exit 2
fi

input="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
binary="${2:-target/release/surroundfold}"
binary="$(cd "$(dirname "$binary")" && pwd)/$(basename "$binary")"
if [[ ! -f "$input" ]]; then
  echo "benchmark input is missing: $input" >&2
  exit 2
fi
if [[ ! -x "$binary" ]]; then
  echo "SurroundFold binary is missing or not executable: $binary" >&2
  exit 2
fi

workspace="$(mktemp -d)"
cleanup() {
  rm -rf -- "$workspace"
}
trap cleanup EXIT

printf 'renderer\trender_seconds\trealtime_multiple\ttotal_seconds\n'
for renderer in baseline continuous-direction image-distance combined; do
  command=(
    "$binary"
    "$input"
    --output "$workspace/$renderer.mkv"
    --progress json
  )
  case "$renderer" in
    continuous-direction)
      command+=(--object-renderer continuous)
      ;;
    image-distance)
      command+=(--distance-renderer image-source)
      ;;
    combined)
      command+=(
        --object-renderer continuous
        --distance-renderer image-source
      )
      ;;
  esac
  result="$("${command[@]}")"
  python3 -c '
import json
import sys

name = sys.argv[1]
result = json.load(sys.stdin)
duration = result["renderedSamples"] / result["sampleRate"]
render = result["timingsSeconds"]["render"]
total = result["timingsSeconds"]["total"]
print(f"{name}\t{render:.3f}\t{duration / render:.2f}\t{total:.3f}")
' "$renderer" <<<"$result"
done
