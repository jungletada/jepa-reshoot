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
INPUT_RESULT_ROOT=${INPUT_RESULT_ROOT:-./results/single}
OUTPUT_RESULT_ROOT=${OUTPUT_RESULT_ROOT:-./results/stitched_single}
FULL_RESULT_BASE=${FULL_RESULT_BASE:-./results/full}
NUM_FRAMES=${NUM_FRAMES:-49}
TRANSLATION_SIGMA=${TRANSLATION_SIGMA:-4.0}
ROTATION_SIGMA=${ROTATION_SIGMA:-4.0}
SMOOTH_MODE=${SMOOTH_MODE:-nearest}
SMOOTHED_CAMERA_NAME=${SMOOTHED_CAMERA_NAME:-cameras_gaussian_smooth.npz}
FORCE_STITCH=${FORCE_STITCH:-false}
OVERWRITE_SPLITS=${OVERWRITE_SPLITS:-false}

if [[ -z "$TARGET" ]]; then
    echo "Usage: bash scripts/test_video/stitch_splits_smooth_and_slice.sh <source_stem|manifest.json>"
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

MANIFEST="$(resolve_manifest "$TARGET")"
SOURCE_STEM="$(basename "$MANIFEST")"
SOURCE_STEM="${SOURCE_STEM%_splits_manifest.json}"
FULL_RECON_FOLDER=${FULL_RECON_FOLDER:-$FULL_RESULT_BASE/${SOURCE_STEM}_stitched_${RESOLUTION}/recon_and_seg}
FULL_SOURCE_CAMERA="$FULL_RECON_FOLDER/cameras.npz"
FULL_SMOOTHED_CAMERA="$FULL_RECON_FOLDER/$SMOOTHED_CAMERA_NAME"

echo "Stitched split kwargs:"
echo "    MANIFEST=$MANIFEST"
echo "    INPUT_RESULT_ROOT=$INPUT_RESULT_ROOT"
echo "    OUTPUT_RESULT_ROOT=$OUTPUT_RESULT_ROOT"
echo "    FULL_RECON_FOLDER=$FULL_RECON_FOLDER"
echo "    RESOLUTION=$RESOLUTION"
echo "    NUM_FRAMES=$NUM_FRAMES"
echo "    TRANSLATION_SIGMA=$TRANSLATION_SIGMA"
echo "    ROTATION_SIGMA=$ROTATION_SIGMA"
echo "    FORCE_STITCH=$FORCE_STITCH"
echo "    OVERWRITE_SPLITS=$OVERWRITE_SPLITS"

STITCH_ARGS=()
if [[ "$FORCE_STITCH" == "true" ]]; then
    STITCH_ARGS+=(--overwrite)
fi
if [[ ! -f "$FULL_SOURCE_CAMERA" || "$FORCE_STITCH" == "true" ]]; then
    python3 -m scripts.preprocess.stitch_split_recon_by_manifest \
        --manifest "$MANIFEST" \
        --input_result_root "$INPUT_RESULT_ROOT" \
        --output_folder "$FULL_RECON_FOLDER" \
        --resolution "$RESOLUTION" \
        --num_frames "$NUM_FRAMES" \
        "${STITCH_ARGS[@]}"
else
    echo "Skip existing stitched reconstruction: $FULL_SOURCE_CAMERA"
fi

python3 -m scripts.preprocess.smooth_camera_trajectory \
    --input "$FULL_SOURCE_CAMERA" \
    --output "$FULL_SMOOTHED_CAMERA" \
    --translation_sigma "$TRANSLATION_SIGMA" \
    --rotation_sigma "$ROTATION_SIGMA" \
    --mode "$SMOOTH_MODE"

SLICE_ARGS=()
if [[ "$OVERWRITE_SPLITS" == "true" ]]; then
    SLICE_ARGS+=(--overwrite)
fi
python3 -m scripts.preprocess.slice_full_recon_by_manifest \
    --manifest "$MANIFEST" \
    --full_recon_and_seg_folder "$FULL_RECON_FOLDER" \
    --smoothed_camera "$FULL_SMOOTHED_CAMERA" \
    --smoothed_camera_name "$SMOOTHED_CAMERA_NAME" \
    --result_root "$OUTPUT_RESULT_ROOT" \
    --resolution "$RESOLUTION" \
    --num_frames "$NUM_FRAMES" \
    "${SLICE_ARGS[@]}"

echo
echo "Stitch, global smoothing, and condition slicing finished."
echo "Full reconstruction: $FULL_RECON_FOLDER"
echo "Per-split conditions: $OUTPUT_RESULT_ROOT"
