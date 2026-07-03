#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

enter_nix_if_needed() {
    if [ -z "${IN_NIX_SHELL:-}" ]; then
        exec "$SCRIPT_DIR/dev.sh" "$0" "$@"
    fi
}

enter_enzyme_nix_if_needed() {
    if [ "${MERCURY_ENZYME_SHELL:-}" != "1" ]; then
        exec nix develop . --command "$0" "$@"
    fi
}

cd_project_root() {
    cd "$PROJECT_ROOT"
}

ensure_logs_dir() {
    mkdir -p "$PROJECT_ROOT/logs"
}
