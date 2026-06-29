#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

if [ -z "${IN_NIX_SHELL:-}" ]; then
    exec nix develop .#perf --command "$0" "$@"
fi

cd_project_root

echo "No memory profiling target is configured yet."
echo "Add a dhat-enabled binary or benchmark before using this script."
