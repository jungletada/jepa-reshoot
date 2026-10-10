#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FORCE_SMOOTH=${FORCE_SMOOTH:-true}
export OVERWRITE_SPLITS=${OVERWRITE_SPLITS:-true}
exec bash "$SCRIPT_DIR/prepare_full_recon_and_slice.sh" "$@"
