#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
enter_enzyme_nix_if_needed "$@"
cd_project_root

BUILD_ARGS=(--release --all-targets --all-features)

for arg in "$@"; do
    case "$arg" in
        --release)
            ;;
        --debug)
            BUILD_ARGS=(--all-targets --all-features)
            ;;
        -h|--help)
            echo "Usage: scripts/build.sh [--debug|--release]"
            exit 0
            ;;
        *)
            echo "Unknown build argument: $arg" >&2
            exit 1
            ;;
    esac
done

cargo build "${BUILD_ARGS[@]}"
