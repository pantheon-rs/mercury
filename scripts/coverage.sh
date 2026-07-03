#!/usr/bin/env bash
set -euo pipefail
# NOTE: Enzyme test legs are cfg(not(coverage))-gated (Enzyme cannot
# differentiate atomic profile counters), so coverage measures the library
# through the non-AD tests; derivative correctness is the normal suite's job.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
enter_nix_if_needed "$@"
cd_project_root

OUT_DIR="build/coverage"
HTML_DIR="$OUT_DIR/html"
LCOV_FILE="$OUT_DIR/lcov.info"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

echo "llvm-cov: ${LLVM_COV:-llvm-cov}"
echo "llvm-profdata: ${LLVM_PROFDATA:-llvm-profdata}"

cargo llvm-cov clean --workspace
cargo llvm-cov --release --all-features --html --output-dir "$OUT_DIR"
cargo llvm-cov --release --all-features --lcov --output-path "$LCOV_FILE"

echo "Coverage HTML: $HTML_DIR/index.html"
echo "Coverage LCOV: $LCOV_FILE"
