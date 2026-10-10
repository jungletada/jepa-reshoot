#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$VISTA4D_ROOT"

VIDEO_PATH=${VIDEO_PATH:-$SOURCE_VIDEO}
OUTPUT_FOLDER=${OUTPUT_FOLDER:-$RECON_AND_SEG_FOLDER}
RECON_METHOD=${RECON_METHOD:-da3}  # da3, pi3
SEG_KEYWORDS=${SEG_KEYWORDS:-"person man woman hand phone bag backpack car"}

PI3_PIXEL_LIMIT=${PI3_PIXEL_LIMIT:-255000}
PI3_HEAD_CHUNK_SIZE=${PI3_HEAD_CHUNK_SIZE:-16}
DA3_PROCESS_RES=${DA3_PROCESS_RES:-672}
DA3_MODEL_ID=${DA3_MODEL_ID:-./checkpoints/DA3NESTED-GIANT-LARGE-1.1}
PI3_MODEL_ID=${PI3_MODEL_ID:-./checkpoints/Pi3X}
SAVE_VIS=${SAVE_VIS:-true}

read -r -a SEG_KEYWORDS_ARRAY <<< "$SEG_KEYWORDS"

ARGS=()
if [[ "$SAVE_VIS" == "true" ]]; then
    ARGS+=(--save_vis)
fi

if [[ ! -f "$VIDEO_PATH" ]]; then
    echo "Video file not found: $VIDEO_PATH"
    exit 1
fi

echo "Script kwargs:"
echo "    SOURCE_VIDEO=$SOURCE_VIDEO"
echo "    EXAMPLE=$EXAMPLE"
echo "    RESULT_ROOT=$RESULT_ROOT"
echo "    VIDEO_PATH=$VIDEO_PATH"
echo "    OUTPUT_FOLDER=$OUTPUT_FOLDER"
echo "    RECON_METHOD=$RECON_METHOD"
echo "    SEG_KEYWORDS=$SEG_KEYWORDS"
echo "    RESOLUTION=$RESOLUTION"
echo "    HEIGHT=$HEIGHT"
echo "    WIDTH=$WIDTH"
echo "    NUM_FRAMES=$NUM_FRAMES"
echo "    PI3_PIXEL_LIMIT=$PI3_PIXEL_LIMIT"
echo "    PI3_HEAD_CHUNK_SIZE=$PI3_HEAD_CHUNK_SIZE"
echo "    DA3_PROCESS_RES=$DA3_PROCESS_RES"
echo "    DA3_MODEL_ID=$DA3_MODEL_ID"
echo "    PI3_MODEL_ID=$PI3_MODEL_ID"
echo "    SAVE_VIS=$SAVE_VIS"

python3 -m scripts.preprocess.recon_and_seg_single \
    --video_path "$VIDEO_PATH" \
    --output_folder "$OUTPUT_FOLDER" \
    --seg_keywords "${SEG_KEYWORDS_ARRAY[@]}" \
    --recon_method "$RECON_METHOD" \
    --da3_model_id "$DA3_MODEL_ID" \
    --pi3_model_id "$PI3_MODEL_ID" \
    --height "$HEIGHT" --width "$WIDTH" --num_frames "$NUM_FRAMES" \
    --pi3_pixel_limit "$PI3_PIXEL_LIMIT" --pi3_head_chunk_size "$PI3_HEAD_CHUNK_SIZE" \
    --da3_process_res "$DA3_PROCESS_RES" \
    "${ARGS[@]}"
