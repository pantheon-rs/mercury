#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
enter_enzyme_nix_if_needed "$@"
cd_project_root
ensure_logs_dir

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="logs/ci_${TIMESTAMP}.log"

{
    echo "=== Mercury CI ==="
    echo "timestamp=$TIMESTAMP"
    echo

    ./scripts/format.sh --check
    cargo clippy --release --all-targets --all-features -- -D warnings
    ./scripts/test.sh
    ./scripts/docs.sh
    ./scripts/audit.sh
} 2>&1 | tee "$LOG_FILE"

ln -sf "ci_${TIMESTAMP}.log" logs/ci.log
echo "CI log: $LOG_FILE"
