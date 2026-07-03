#!/usr/bin/env bash
# Verifies the crate version was bumped relative to a baseline ref and that
# Cargo.lock agrees with Cargo.toml. Pure git/grep — no toolchain needed, so
# CI can run it without entering the nix shell.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

BASELINE="${1:-origin/main}"

current="$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)"
baseline_version="$(git show "$BASELINE:Cargo.toml" 2>/dev/null | grep -m1 '^version' | cut -d'"' -f2 || true)"

if [ -z "$baseline_version" ]; then
    echo "No Cargo.toml at $BASELINE; skipping version-bump check."
    exit 0
fi

if [ "$current" = "$baseline_version" ]; then
    echo "ERROR: version is still $current (same as $BASELINE)." >&2
    echo "Bump it before merging: ./scripts/bump-version.sh <major|minor|patch>" >&2
    exit 1
fi

highest="$(printf '%s\n%s\n' "$baseline_version" "$current" | sort -V | tail -1)"
if [ "$highest" != "$current" ]; then
    echo "ERROR: version $current is LOWER than baseline $baseline_version at $BASELINE." >&2
    exit 1
fi

lock_version="$(grep -A1 '^name = "mercury"' Cargo.lock | grep -m1 version | cut -d'"' -f2)"
if [ "$lock_version" != "$current" ]; then
    echo "ERROR: Cargo.lock has mercury $lock_version but Cargo.toml says $current." >&2
    echo "Run ./scripts/bump-version.sh (or cargo update -w) and commit Cargo.lock." >&2
    exit 1
fi

echo "Version bump OK: $baseline_version -> $current (Cargo.lock in sync)."
