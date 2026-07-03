#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
enter_nix_if_needed "$@"
cd_project_root

./scripts/ci.sh
cargo package --allow-dirty

# Semver-check the public API against the merge baseline, NOT crates.io:
# the bare `mercury` name on crates.io is an unrelated crate (this one is
# publish = false), so the default registry baseline is meaningless.
BASELINE="${MERCURY_SEMVER_BASELINE:-origin/main}"
if git rev-parse --verify --quiet "$BASELINE" >/dev/null; then
    cargo semver-checks check-release --baseline-rev "$BASELINE"
else
    echo "WARNING: baseline $BASELINE not found; skipping semver-checks"
fi

./scripts/check-version-bump.sh
