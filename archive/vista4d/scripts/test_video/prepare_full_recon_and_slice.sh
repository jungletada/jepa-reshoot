#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISTA4D_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$VISTA4D_ROOT"

TARGET=${1:-${SPLIT_TARGET:-}}
RESOLUTION=${RESOLUTION:-384p}
if [[ -z "${SPLITS_DIR+x}" ]]; then
    if [[ "$RESOLUTION" == "720p" ]]; then
        SPLITS_DIR=./media/splits/720p
    else
        SPLITS_DIR=./media/splits
    fi
fi
FULL_MEDIA_DIR=${FULL_MEDIA_DIR:-./media/full}
FULL_RESULT_BASE=${FULL_RESULT_BASE:-./results/full}
RESULT_ROOT=${RESULT_ROOT:-./results/single}
NUM_FRAMES=${NUM_FRAMES:-49}
RECON_METHOD=${RECON_METHOD:-da3}
SEG_KEYWORDS=${SEG_KEYWORDS:-"person man woman hand phone bag backpack car stroller"}
PI3_MODEL_ID=${PI3_MODEL_ID:-./checkpoints/Pi3X}
DA3_MODEL_ID=${DA3_MODEL_ID:-./checkpoints/DA3NESTED-GIANT-LARGE-1.1}
PI3_PIXEL_LIMIT=${PI3_PIXEL_LIMIT:-255000}
PI3_HEAD_CHUNK_SIZE=${PI3_HEAD_CHUNK_SIZE:-16}
DA3_PROCESS_RES=${DA3_PROCESS_RES:-448}
TRANSLATION_SIGMA=${TRANSLATION_SIGMA:-4.0}
ROTATION_SIGMA=${ROTATION_SIGMA:-4.0}
SMOOTH_MODE=${SMOOTH_MODE:-nearest}
SMOOTHED_CAMERA_NAME=${SMOOTHED_CAMERA_NAME:-cameras_gaussian_smooth.npz}
SAVE_VIS_FULL=${SAVE_VIS_FULL:-false}
FORCE=${FORCE:-false}
FORCE_PREPARED_VIDEO=${FORCE_PREPARED_VIDEO:-$FORCE}
FORCE_RECON=${FORCE_RECON:-$FORCE}
FORCE_SMOOTH=${FORCE_SMOOTH:-$FORCE}
OVERWRITE_SPLITS=${OVERWRITE_SPLITS:-$FORCE}

if [[ -z "$TARGET" ]]; then
    echo "Usage: bash scripts/test_video/prepare_full_recon_and_slice.sh <source_stem|manifest.json>"
    echo
    echo "This runs reconstruction/SAM3 once on the full manifest range, smooths the full camera"
    echo "path, then slices conditions into the existing results/single/<clip>/recon_and_seg layout."
    exit 1
fi

if [[ "$RESOLUTION" == "384p" ]]; then
    EXPECTED_HEIGHT=384
    EXPECTED_WIDTH=672
elif [[ "$RESOLUTION" == "720p" ]]; then
    EXPECTED_HEIGHT=720
    EXPECTED_WIDTH=1280
else
    echo "Unrecognized RESOLUTION=$RESOLUTION, expected 384p or 720p."
    exit 1
fi
HEIGHT=${HEIGHT:-$EXPECTED_HEIGHT}
WIDTH=${WIDTH:-$EXPECTED_WIDTH}
if [[ "$HEIGHT" != "$EXPECTED_HEIGHT" || "$WIDTH" != "$EXPECTED_WIDTH" ]]; then
    echo "RESOLUTION=$RESOLUTION requires WIDTH x HEIGHT " \
        "$EXPECTED_WIDTH x $EXPECTED_HEIGHT, got $WIDTH x $HEIGHT." >&2
    exit 2
fi

resolve_manifest() {
    local target="$1"
    if [[ -f "$target" && "$target" == *.json ]]; then
        echo "$target"
        return
    fi

    local base
    base="$(basename "$target")"
    base="${base%.*}"
    base="${base%_splits_manifest}"
    local manifest="$SPLITS_DIR/${base}_splits_manifest.json"
    if [[ ! -f "$manifest" ]]; then
        echo "Could not find JSON split manifest: $manifest" >&2
        exit 1
    fi
    echo "$manifest"
}

MANIFEST="$(resolve_manifest "$TARGET")"
mapfile -t MANIFEST_META < <(
    python3 - "$MANIFEST" <<'PY'
import json
from pathlib import Path
import sys

with Path(sys.argv[1]).open("r", encoding="utf-8") as file:
    data = json.load(file)
clips = data["clips"]
start = int(data.get("start_frame", min(int(clip["start_frame"]) for clip in clips)))
end = int(data.get("end_exclusive", max(int(clip["end_frame"]) for clip in clips) + 1))
print(data["input_path"])
print(start)
print(end)
PY
)

MANIFEST_SOURCE_VIDEO="${MANIFEST_META[0]}"
SOURCE_VIDEO=${SOURCE_VIDEO:-$MANIFEST_SOURCE_VIDEO}
if [[ ! -f "$SOURCE_VIDEO" && -f "./data/$(basename "$SOURCE_VIDEO")" ]]; then
    SOURCE_VIDEO="./data/$(basename "$SOURCE_VIDEO")"
fi
if [[ ! -f "$SOURCE_VIDEO" ]]; then
    echo "Source video not found: $SOURCE_VIDEO" >&2
    echo "Set SOURCE_VIDEO to the local path for this machine." >&2
    exit 1
fi
FULL_START_FRAME="${MANIFEST_META[1]}"
FULL_END_EXCLUSIVE="${MANIFEST_META[2]}"
FULL_NUM_FRAMES=$((FULL_END_EXCLUSIVE - FULL_START_FRAME))
FULL_END_FRAME=$((FULL_END_EXCLUSIVE - 1))
SOURCE_STEM="$(basename "$SOURCE_VIDEO")"
SOURCE_STEM="${SOURCE_STEM%.*}"
FULL_EXAMPLE="${SOURCE_STEM}_frames$(printf '%06d' "$FULL_START_FRAME")_$(printf '%06d' "$FULL_END_FRAME")_${RESOLUTION}${FULL_NUM_FRAMES}"
FULL_VIDEO=${FULL_VIDEO:-$FULL_MEDIA_DIR/$FULL_EXAMPLE.mp4}
FULL_RESULT_EXAMPLE="${FULL_EXAMPLE}_${RECON_METHOD}"
FULL_RESULT_ROOT=${FULL_RESULT_ROOT:-$FULL_RESULT_BASE/$FULL_RESULT_EXAMPLE}
FULL_RECON_FOLDER=${FULL_RECON_FOLDER:-$FULL_RESULT_ROOT/recon_and_seg}
FULL_SOURCE_CAMERA=${FULL_SOURCE_CAMERA:-$FULL_RECON_FOLDER/cameras.npz}
FULL_SMOOTHED_CAMERA=${FULL_SMOOTHED_CAMERA:-$FULL_RECON_FOLDER/$SMOOTHED_CAMERA_NAME}

read -r -a SEG_KEYWORDS_ARRAY <<< "$SEG_KEYWORDS"

echo "Full-sequence condition kwargs:"
echo "    MANIFEST=$MANIFEST"
echo "    SOURCE_VIDEO=$SOURCE_VIDEO"
echo "    FULL_RANGE=$FULL_START_FRAME..$FULL_END_FRAME ($FULL_NUM_FRAMES frames)"
echo "    FULL_VIDEO=$FULL_VIDEO"
echo "    FULL_RECON_FOLDER=$FULL_RECON_FOLDER"
echo "    RESULT_ROOT=$RESULT_ROOT"
echo "    RECON_METHOD=$RECON_METHOD"
echo "    SEG_KEYWORDS=$SEG_KEYWORDS"
echo "    RESOLUTION=$RESOLUTION ($WIDTH x $HEIGHT)"
echo "    PI3_PIXEL_LIMIT=$PI3_PIXEL_LIMIT"
echo "    PI3_HEAD_CHUNK_SIZE=$PI3_HEAD_CHUNK_SIZE"
echo "    DA3_PROCESS_RES=$DA3_PROCESS_RES"
echo "    TRANSLATION_SIGMA=$TRANSLATION_SIGMA"
echo "    ROTATION_SIGMA=$ROTATION_SIGMA"
echo "    OVERWRITE_SPLITS=$OVERWRITE_SPLITS"

if [[ ! -f "$FULL_VIDEO" || "$FORCE_PREPARED_VIDEO" == "true" ]]; then
    python3 -m scripts.preprocess.prepare_custom_single_video \
        --input "$SOURCE_VIDEO" \
        --output_dir "$(dirname "$FULL_VIDEO")" \
        --output_name "$(basename "${FULL_VIDEO%.mp4}")" \
        --start_frame "$FULL_START_FRAME" \
        --num_frames "$FULL_NUM_FRAMES" \
        --height "$HEIGHT" --width "$WIDTH"
else
    echo "Skip existing prepared full video: $FULL_VIDEO"
fi

RECON_ARGS=()
if [[ "$SAVE_VIS_FULL" == "true" ]]; then
    RECON_ARGS+=(--save_vis)
fi
if [[ ! -f "$FULL_SOURCE_CAMERA" || "$FORCE_RECON" == "true" ]]; then
    python3 -m scripts.preprocess.recon_and_seg_single \
        --video_path "$FULL_VIDEO" \
        --output_folder "$FULL_RECON_FOLDER" \
        --seg_keywords "${SEG_KEYWORDS_ARRAY[@]}" \
        --recon_method "$RECON_METHOD" \
        --pi3_model_id "$PI3_MODEL_ID" \
        --da3_model_id "$DA3_MODEL_ID" \
        --height "$HEIGHT" --width "$WIDTH" --num_frames "$FULL_NUM_FRAMES" \
        --pi3_pixel_limit "$PI3_PIXEL_LIMIT" --pi3_head_chunk_size "$PI3_HEAD_CHUNK_SIZE" \
        --da3_process_res "$DA3_PROCESS_RES" \
        "${RECON_ARGS[@]}"
else
    echo "Skip existing full reconstruction: $FULL_SOURCE_CAMERA"
fi

if [[ ! -f "$FULL_SMOOTHED_CAMERA" || "$FORCE_SMOOTH" == "true" ]]; then
    python3 -m scripts.preprocess.smooth_camera_trajectory \
        --input "$FULL_SOURCE_CAMERA" \
        --output "$FULL_SMOOTHED_CAMERA" \
        --translation_sigma "$TRANSLATION_SIGMA" \
        --rotation_sigma "$ROTATION_SIGMA" \
        --mode "$SMOOTH_MODE"
else
    echo "Skip existing full smoothed camera: $FULL_SMOOTHED_CAMERA"
fi

SLICE_ARGS=()
if [[ "$OVERWRITE_SPLITS" == "true" ]]; then
    SLICE_ARGS+=(--overwrite)
fi
python3 -m scripts.preprocess.slice_full_recon_by_manifest \
    --manifest "$MANIFEST" \
    --full_recon_and_seg_folder "$FULL_RECON_FOLDER" \
    --smoothed_camera "$FULL_SMOOTHED_CAMERA" \
    --smoothed_camera_name "$SMOOTHED_CAMERA_NAME" \
    --result_root "$RESULT_ROOT" \
    --resolution "$RESOLUTION" \
    --num_frames "$NUM_FRAMES" \
    "${SLICE_ARGS[@]}"

echo
echo "Full-sequence reconstruction and condition slicing finished."
echo "Next: bash scripts/test_video/run_splits_render.sh $TARGET"
