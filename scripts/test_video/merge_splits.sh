#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISTA4D_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$VISTA4D_ROOT"

TARGET=${1:-${SPLIT_TARGET:-1778135019043}}
SPLITS_DIR=${SPLITS_DIR:-./media/splits}
RESOLUTION=${RESOLUTION:-384p}
NUM_FRAMES=${NUM_FRAMES:-49}
USE_SMOOTHED_CAMERA=${USE_SMOOTHED_CAMERA:-true}
MERGE_MODE=${MERGE_MODE:-center_cut}
FLOW_MAX_PIXELS=${FLOW_MAX_PIXELS:-32}
FLOW_CONSISTENCY_SIGMA=${FLOW_CONSISTENCY_SIGMA:-2.5}
MERGE_SEED=${MERGE_SEED:-${SEEDS:-10027}}
MERGE_SEED="${MERGE_SEED%% *}"
RESULT_ROOT=${RESULT_ROOT:-./results/single}
MERGED_OUTPUT_DIR=${MERGED_OUTPUT_DIR:-./results/merged}
VIDEO_NAME=${VIDEO_NAME:-auto}

derive_source_stem() {
    local target="$1"
    local base
    base="$(basename "$target")"
    base="${base%.*}"
    base="${base%_splits_manifest}"
    echo "$base"
}

resolve_manifest() {
    local target="$1"
    if [[ -f "$target" ]]; then
        echo "$target"
        return
    fi

    local stem
    stem="$(derive_source_stem "$target")"
    local csv_manifest="$SPLITS_DIR/${stem}_splits_manifest.csv"
    local json_manifest="$SPLITS_DIR/${stem}_splits_manifest.json"
    if [[ -f "$csv_manifest" ]]; then
        echo "$csv_manifest"
        return
    fi
    if [[ -f "$json_manifest" ]]; then
        echo "$json_manifest"
        return
    fi

    echo "Could not find split manifest for target: $target" >&2
    echo "Searched: $csv_manifest" >&2
    echo "Searched: $json_manifest" >&2
    exit 1
}

if [[ -z "${INFERENCE_FOLDER:-}" ]]; then
    if [[ "$USE_SMOOTHED_CAMERA" == "true" ]]; then
        INFERENCE_FOLDER="vista4d_${RESOLUTION}_smooth"
    else
        INFERENCE_FOLDER="vista4d_$RESOLUTION"
    fi
fi

MANIFEST="$(resolve_manifest "$TARGET")"
SOURCE_STEM="$(derive_source_stem "$MANIFEST")"
OUTPUT=${OUTPUT:-$MERGED_OUTPUT_DIR/${SOURCE_STEM}_${INFERENCE_FOLDER}_seed=${MERGE_SEED}_${MERGE_MODE}.mp4}

echo "Merge kwargs:"
echo "    TARGET=$TARGET"
echo "    MANIFEST=$MANIFEST"
echo "    RESULT_ROOT=$RESULT_ROOT"
echo "    RESOLUTION=$RESOLUTION"
echo "    NUM_FRAMES=$NUM_FRAMES"
echo "    INFERENCE_FOLDER=$INFERENCE_FOLDER"
echo "    VIDEO_NAME=$VIDEO_NAME"
echo "    MERGE_SEED=$MERGE_SEED"
echo "    MERGE_MODE=$MERGE_MODE"
echo "    FLOW_MAX_PIXELS=$FLOW_MAX_PIXELS"
echo "    FLOW_CONSISTENCY_SIGMA=$FLOW_CONSISTENCY_SIGMA"
echo "    OUTPUT=$OUTPUT"

python3 -m scripts.postprocess.merge_split_videos \
    --manifest "$MANIFEST" \
    --result_root "$RESULT_ROOT" \
    --resolution "$RESOLUTION" \
    --num_frames "$NUM_FRAMES" \
    --inference_folder "$INFERENCE_FOLDER" \
    --video_name "$VIDEO_NAME" \
    --seed "$MERGE_SEED" \
    --merge_mode "$MERGE_MODE" \
    --flow_max_pixels "$FLOW_MAX_PIXELS" \
    --flow_consistency_sigma "$FLOW_CONSISTENCY_SIGMA" \
    --output "$OUTPUT"
