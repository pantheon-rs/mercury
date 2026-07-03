#!/usr/bin/env bash
# Bumps the crate version — the SINGLE source of truth is Cargo.toml
# (nix/packages.nix reads it at eval time; Cargo.lock is synced here).
#
# Usage: scripts/bump-version.sh <major|minor|patch|X.Y.Z>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
enter_nix_if_needed "$@"
cd_project_root

if [ $# -ne 1 ]; then
    echo "Usage: $0 <major|minor|patch|X.Y.Z>" >&2
    exit 1
fi

current="$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)"
IFS='.' read -r major minor patch <<< "$current"

case "$1" in
    major) new="$((major + 1)).0.0" ;;
    minor) new="$major.$((minor + 1)).0" ;;
    patch) new="$major.$minor.$((patch + 1))" ;;
    *)
        if [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            new="$1"
        else
            echo "ERROR: '$1' is not major|minor|patch or X.Y.Z" >&2
            exit 1
        fi
        ;;
esac

sed -i "0,/^version = \"$current\"/s//version = \"$new\"/" Cargo.toml
cargo update --workspace --quiet   # sync Cargo.lock

echo "Bumped: $current -> $new (Cargo.toml + Cargo.lock)"
echo "Review and commit:"
echo "  git add Cargo.toml Cargo.lock && git commit -m \"chore: bump version to $new\""
