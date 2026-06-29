#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
enter_nix_if_needed "$@"
cd_project_root

OUT_DIR="build/coverage"
HTML_DIR="$OUT_DIR/html"
LCOV_FILE="$OUT_DIR/lcov.info"

mkdir -p "$OUT_DIR"

cargo llvm-cov clean --workspace
cargo llvm-cov --all-features --html --output-dir "$HTML_DIR"
cargo llvm-cov --all-features --lcov --output-path "$LCOV_FILE"

echo "Coverage HTML: $HTML_DIR/index.html"
echo "Coverage LCOV: $LCOV_FILE"
