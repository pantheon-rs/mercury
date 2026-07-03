#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
    exec nix develop .
fi

exec nix develop . --command "$@"
