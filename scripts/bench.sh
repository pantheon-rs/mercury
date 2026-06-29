#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
enter_nix_if_needed "$@"
cd_project_root

if [ ! -d benches ] || ! find benches -name '*.rs' -print -quit | grep -q .; then
    echo "No benchmarks found in benches/."
    exit 0
fi

cargo bench --all-features "$@"
