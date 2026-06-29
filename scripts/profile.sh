#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

if [ -z "${IN_NIX_SHELL:-}" ]; then
    exec nix develop .#perf --command "$0" "$@"
fi

cd_project_root

if [ "$#" -eq 0 ]; then
    echo "Usage: scripts/profile.sh <cargo-flamegraph-args>" >&2
    echo "Example: scripts/profile.sh --bench scalar_ops" >&2
    exit 1
fi

cargo flamegraph "$@"
