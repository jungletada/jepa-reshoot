#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$VISTA4D_ROOT"

TRANSLATION_SIGMA=${TRANSLATION_SIGMA:-4.0}
ROTATION_SIGMA=${ROTATION_SIGMA:-4.0}
SMOOTH_MODE=${SMOOTH_MODE:-nearest}
ANCHOR_FIRST=${ANCHOR_FIRST:-true}

ARGS=()
if [[ "$ANCHOR_FIRST" != "true" ]]; then
    ARGS+=(--no_anchor_first)
fi

if [[ ! -f "$SOURCE_CAMERA_PATH" ]]; then
    echo "Source camera file not found: $SOURCE_CAMERA_PATH"
    echo "Run scripts/test_video/recon_and_seg.sh first, or override SOURCE_CAMERA_PATH."
    exit 1
fi

echo "Script kwargs:"
echo "    SOURCE_VIDEO=$SOURCE_VIDEO"
echo "    EXAMPLE=$EXAMPLE"
echo "    SOURCE_CAMERA_PATH=$SOURCE_CAMERA_PATH"
echo "    SMOOTHED_CAMERA_PATH=$SMOOTHED_CAMERA_PATH"
echo "    TRANSLATION_SIGMA=$TRANSLATION_SIGMA"
echo "    ROTATION_SIGMA=$ROTATION_SIGMA"
echo "    SMOOTH_MODE=$SMOOTH_MODE"
echo "    ANCHOR_FIRST=$ANCHOR_FIRST"

python3 -m scripts.preprocess.smooth_camera_trajectory \
    --input "$SOURCE_CAMERA_PATH" \
    --output "$SMOOTHED_CAMERA_PATH" \
    --translation_sigma "$TRANSLATION_SIGMA" \
    --rotation_sigma "$ROTATION_SIGMA" \
    --mode "$SMOOTH_MODE" \
    "${ARGS[@]}"
