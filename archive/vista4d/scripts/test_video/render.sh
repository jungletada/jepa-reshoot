#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$VISTA4D_ROOT"

SAVE_VIS=${SAVE_VIS:-true}
RENDER_ONLY_NECESSARY=${RENDER_ONLY_NECESSARY:-true}

ARGS=()
if [[ "$SAVE_VIS" == "true" ]]; then
    ARGS+=(--save_vis)
fi
if [[ "$RENDER_ONLY_NECESSARY" == "true" ]]; then
    ARGS+=(--render_only_necessary)
fi

if [[ ! -d "$RECON_AND_SEG_FOLDER" ]]; then
    echo "Reconstruction folder not found: $RECON_AND_SEG_FOLDER"
    echo "Run scripts/test_video/recon_and_seg.sh first, or override RECON_AND_SEG_FOLDER."
    exit 1
fi
if [[ ! -f "$CAM_PATH" ]]; then
    echo "Camera path not found: $CAM_PATH"
    echo "Run scripts/test_video/smooth.sh first, set USE_SMOOTHED_CAMERA=false, or override CAM_PATH."
    exit 1
fi

echo "Script kwargs:"
echo "    SOURCE_VIDEO=$SOURCE_VIDEO"
echo "    EXAMPLE=$EXAMPLE"
echo "    RECON_AND_SEG_FOLDER=$RECON_AND_SEG_FOLDER"
echo "    CAM_PATH=$CAM_PATH"
echo "    RENDER_OUTPUT_FOLDER=$RENDER_OUTPUT_FOLDER"
echo "    RESOLUTION=$RESOLUTION"
echo "    HEIGHT=$HEIGHT"
echo "    WIDTH=$WIDTH"
echo "    NUM_FRAMES=$NUM_FRAMES"
echo "    SAVE_VIS=$SAVE_VIS"
echo "    RENDER_ONLY_NECESSARY=$RENDER_ONLY_NECESSARY"

python3 -m scripts.preprocess.render_single \
    --recon_and_seg_folder "$RECON_AND_SEG_FOLDER" \
    --cam_path "$CAM_PATH" \
    --output_folder "$RENDER_OUTPUT_FOLDER" \
    --height "$HEIGHT" --width "$WIDTH" --num_frames "$NUM_FRAMES" \
    "${ARGS[@]}"
