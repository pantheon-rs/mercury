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

    CLIPPY_LOG="$(mktemp)"
    if ! cargo clippy --release --all-targets --all-features -- -D warnings \
        >"$CLIPPY_LOG" 2>&1; then
        if grep -q "autodiff backend not found in the sysroot: failed to find a \`libEnzyme-22\` folder" \
            "$CLIPPY_LOG"; then
            cat "$CLIPPY_LOG"
            echo
            echo "WARNING: clippy skipped — clippy-driver lacks the Enzyme sysroot" \
                "(known toolchain gap, see" \
                "docs/implementation-plans/phase-2-core-types-and-linalg.md);" \
                "lints have NOT been evaluated"
        else
            cat "$CLIPPY_LOG"
            rm -f "$CLIPPY_LOG"
            exit 1
        fi
    else
        cat "$CLIPPY_LOG"
    fi
    rm -f "$CLIPPY_LOG"

    ./scripts/test.sh
    ./scripts/docs.sh
    ./scripts/audit.sh
} 2>&1 | tee "$LOG_FILE"

ln -sf "ci_${TIMESTAMP}.log" logs/ci.log
echo "CI log: $LOG_FILE"
