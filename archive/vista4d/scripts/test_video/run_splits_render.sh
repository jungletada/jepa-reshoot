#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEPS="render" bash "$SCRIPT_DIR/run_splits_pipeline.sh" "$@"
