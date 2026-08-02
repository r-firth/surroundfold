#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "universal macOS packaging must run on macOS" >&2
  exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
release_root="$repository_root/artifacts/release"
arm_target="aarch64-apple-darwin"
intel_target="x86_64-apple-darwin"
package_root="$release_root/surroundfold-macos-universal"
archive="$release_root/surroundfold-macos-universal.tar.gz"
flag_separator=$'\x1f'
build_root="${HOME:?HOME must be set}"
remap_flags="--remap-path-prefix=$build_root=/build${flag_separator}--remap-path-prefix=$repository_root=/source"

rustup target add "$arm_target" "$intel_target"
CARGO_ENCODED_RUSTFLAGS="$remap_flags" cargo build \
  --manifest-path "$repository_root/Cargo.toml" \
  --release \
  --locked \
  --target "$arm_target"
CARGO_ENCODED_RUSTFLAGS="$remap_flags" cargo build \
  --manifest-path "$repository_root/Cargo.toml" \
  --release \
  --locked \
  --target "$intel_target"

mkdir -p "$package_root"
lipo -create \
  "$repository_root/target/$arm_target/release/surroundfold" \
  "$repository_root/target/$intel_target/release/surroundfold" \
  -output "$package_root/surroundfold"
chmod 755 "$package_root/surroundfold"
codesign --force --sign - "$package_root/surroundfold"
cp "$repository_root/README.md" "$package_root/README.md"
cp "$repository_root/CHANGELOG.md" "$package_root/CHANGELOG.md"

tar -C "$release_root" -czf "$archive" "$(basename "$package_root")"
echo "$archive"
