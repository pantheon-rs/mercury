#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
    exec nix develop .#enzyme
fi

exec nix develop .#enzyme --command "$@"
