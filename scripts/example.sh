#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

case "${1:-}" in
    --help|-h|--list)
        if [[ $# != 1 ]]; then
            echo "Unexpected argument: $2" >&2
            exit 2
        fi
        ;;
    -*)
        echo "Unknown option: $1 (use --help)" >&2
        exit 2
        ;;
esac

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: scripts/example.sh [--list | NAME [-- ARGS...]]"
    echo "Lists examples by default. Runs in the pinned Enzyme shell with release mode."
    exit 0
fi

enter_enzyme_nix_if_needed "$@"
cd_project_root

if [[ $# == 0 || "$1" == "--list" ]]; then
    cargo metadata --no-deps --format-version 1 --locked |
        jq -r '[.packages[] | select(.name == "mercury") | .targets[] |
            select(.kind | index("example")) | .name] | sort[]'
    exit 0
fi

example="$1"
shift
if [[ "${1:-}" == "--" ]]; then
    shift
fi

exec cargo run --release --locked --package mercury --example "$example" -- "$@"
