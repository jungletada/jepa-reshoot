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
FULL_RESULT_BASE=${FULL_RESULT_BASE:-./results/full}
OUTPUT_RESULT_ROOT=${OUTPUT_RESULT_ROOT:-./results/shared_static_single}
NUM_FRAMES=${NUM_FRAMES:-49}
STATIC_FRAME_STRIDE=${STATIC_FRAME_STRIDE:-4}
RENDER_CHUNK_SIZE=${RENDER_CHUNK_SIZE:-4}
DEPTH_OUTLIERS=${DEPTH_OUTLIERS:-gaussian}
OVERWRITE_FULL_RENDER=${OVERWRITE_FULL_RENDER:-false}
OVERWRITE_SPLITS=${OVERWRITE_SPLITS:-false}

if [[ -z "$TARGET" ]]; then
    echo "Usage: bash scripts/test_video/render_shared_static_and_slice.sh <source_stem|manifest.json>"
    exit 1
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

MANIFEST="$(resolve_manifest "$TARGET")"
SOURCE_STEM="$(basename "$MANIFEST")"
SOURCE_STEM="${SOURCE_STEM%_splits_manifest.json}"
FULL_SEQUENCE_ROOT=${FULL_SEQUENCE_ROOT:-$FULL_RESULT_BASE/${SOURCE_STEM}_stitched_${RESOLUTION}}
FULL_RECON_FOLDER=${FULL_RECON_FOLDER:-$FULL_SEQUENCE_ROOT/recon_and_seg}
CAM_PATH=${CAM_PATH:-$FULL_RECON_FOLDER/cameras_gaussian_smooth.npz}
FULL_RENDER_FOLDER=${FULL_RENDER_FOLDER:-$FULL_SEQUENCE_ROOT/render_${RESOLUTION}_smooth_shared_static}
RENDER_FOLDER_NAME=${RENDER_FOLDER_NAME:-render_${RESOLUTION}_smooth}

if [[ ! -d "$FULL_RECON_FOLDER" ]]; then
    echo "Full stitched reconstruction not found: $FULL_RECON_FOLDER" >&2
    exit 1
fi
if [[ ! -f "$CAM_PATH" ]]; then
    echo "Full smoothed camera not found: $CAM_PATH" >&2
    exit 1
fi

echo "Shared-static full render kwargs:"
echo "    MANIFEST=$MANIFEST"
echo "    FULL_RECON_FOLDER=$FULL_RECON_FOLDER"
echo "    CAM_PATH=$CAM_PATH"
echo "    FULL_RENDER_FOLDER=$FULL_RENDER_FOLDER"
echo "    OUTPUT_RESULT_ROOT=$OUTPUT_RESULT_ROOT"
echo "    RENDER_FOLDER_NAME=$RENDER_FOLDER_NAME"
echo "    RESOLUTION=$RESOLUTION"
echo "    HEIGHT=$HEIGHT"
echo "    WIDTH=$WIDTH"
echo "    NUM_FRAMES=$NUM_FRAMES"
echo "    STATIC_FRAME_STRIDE=$STATIC_FRAME_STRIDE"
echo "    RENDER_CHUNK_SIZE=$RENDER_CHUNK_SIZE"
echo "    DEPTH_OUTLIERS=$DEPTH_OUTLIERS"
echo "    OVERWRITE_FULL_RENDER=$OVERWRITE_FULL_RENDER"
echo "    OVERWRITE_SPLITS=$OVERWRITE_SPLITS"

RENDER_ARGS=()
if [[ "$OVERWRITE_FULL_RENDER" == "true" ]]; then
    RENDER_ARGS+=(--overwrite)
fi
if [[ ! -f "$FULL_RENDER_FOLDER/shared_static_render.json" || "$OVERWRITE_FULL_RENDER" == "true" ]]; then
    python3 -m scripts.preprocess.render_full_shared_static \
        --recon_and_seg_folder "$FULL_RECON_FOLDER" \
        --cam_path "$CAM_PATH" \
        --output_folder "$FULL_RENDER_FOLDER" \
        --height "$HEIGHT" \
        --width "$WIDTH" \
        --static_frame_stride "$STATIC_FRAME_STRIDE" \
        --render_chunk_size "$RENDER_CHUNK_SIZE" \
        --depth_outliers "$DEPTH_OUTLIERS" \
        "${RENDER_ARGS[@]}"
else
    echo "Skip existing full shared-static render: $FULL_RENDER_FOLDER"
fi

SLICE_ARGS=()
if [[ "$OVERWRITE_SPLITS" == "true" ]]; then
    SLICE_ARGS+=(--overwrite)
fi
python3 -m scripts.preprocess.slice_full_render_by_manifest \
    --manifest "$MANIFEST" \
    --full_render_folder "$FULL_RENDER_FOLDER" \
    --result_root "$OUTPUT_RESULT_ROOT" \
    --render_folder_name "$RENDER_FOLDER_NAME" \
    --resolution "$RESOLUTION" \
    --num_frames "$NUM_FRAMES" \
    "${SLICE_ARGS[@]}"

echo
echo "Shared-static full rendering and condition slicing finished."
echo "Full render: $FULL_RENDER_FOLDER"
echo "Per-split renders: $OUTPUT_RESULT_ROOT"
