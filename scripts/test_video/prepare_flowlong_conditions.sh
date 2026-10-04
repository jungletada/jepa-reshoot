#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISTA4D_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$VISTA4D_ROOT"

TARGET=${1:-${SPLIT_TARGET:-}}
RESOLUTION=${RESOLUTION:-384p}
if [[ -z "${SPLITS_DIR+x}" ]]; then
    if [[ "$RESOLUTION" == "720p" ]]; then
        SPLITS_DIR=./media/flowlong_splits/720p
    else
        SPLITS_DIR=./media/flowlong_splits
    fi
fi
FULL_RESULT_BASE=${FULL_RESULT_BASE:-./results/full}
OUTPUT_RESULT_ROOT=${OUTPUT_RESULT_ROOT:-./results/flowlong_single}
NUM_FRAMES=${NUM_FRAMES:-49}
EXPECTED_OVERLAP=${EXPECTED_OVERLAP:-25}
EXPECTED_STRIDE=${EXPECTED_STRIDE:-24}
TEMPORAL_ALIGNMENT=${TEMPORAL_ALIGNMENT:-4}
STATIC_FRAME_STRIDE=${STATIC_FRAME_STRIDE:-4}
RENDER_CHUNK_SIZE=${RENDER_CHUNK_SIZE:-4}
DEPTH_OUTLIERS=${DEPTH_OUTLIERS:-gaussian}
OVERWRITE_FULL_RENDER=${OVERWRITE_FULL_RENDER:-false}
OVERWRITE_SPLITS=${OVERWRITE_SPLITS:-false}

if [[ -z "$TARGET" ]]; then
    echo "Usage: bash scripts/test_video/prepare_flowlong_conditions.sh <source_stem|manifest.json>"
    echo
    echo "Generates per-window conditions from one full shared-static render."
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
        echo "Could not find FlowLong JSON split manifest: $manifest" >&2
        exit 1
    fi
    echo "$manifest"
}

MANIFEST="$(resolve_manifest "$TARGET")"
SOURCE_STEM="$(basename "$MANIFEST")"
SOURCE_STEM="${SOURCE_STEM%_splits_manifest.json}"
RENDER_FOLDER=${RENDER_FOLDER:-render_${RESOLUTION}_smooth}

python3 - "$MANIFEST" "$NUM_FRAMES" "$EXPECTED_OVERLAP" "$EXPECTED_STRIDE" "$TEMPORAL_ALIGNMENT" <<'PY'
import json
from pathlib import Path
import sys

from utils.split_manifest import validate_flowlong_window_manifest

manifest_path = Path(sys.argv[1])
expected_frames = int(sys.argv[2])
expected_overlap = int(sys.argv[3])
expected_stride = int(sys.argv[4])
alignment = int(sys.argv[5])
if expected_frames - expected_overlap != expected_stride:
    raise ValueError("Expected window geometry is internally inconsistent")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
clips = validate_flowlong_window_manifest(
    manifest,
    clip_frames=expected_frames,
    overlap=expected_overlap,
    temporal_alignment=alignment,
)
starts = [int(clip["start_frame"]) for clip in clips]
print(
    "FlowLong manifest check: PASS "
    f"({len(clips)} windows, starts={starts}, overlap={expected_overlap}, "
    f"stride={expected_stride})"
)
PY

echo "FlowLong shared-condition kwargs:"
echo "    MANIFEST=$MANIFEST"
echo "    FULL_RESULT_BASE=$FULL_RESULT_BASE"
echo "    OUTPUT_RESULT_ROOT=$OUTPUT_RESULT_ROOT"
echo "    RENDER_FOLDER=$RENDER_FOLDER"
echo "    RESOLUTION=$RESOLUTION"
echo "    NUM_FRAMES=$NUM_FRAMES"
echo "    STATIC_FRAME_STRIDE=$STATIC_FRAME_STRIDE"
echo "    RENDER_CHUNK_SIZE=$RENDER_CHUNK_SIZE"
echo "    OVERWRITE_FULL_RENDER=$OVERWRITE_FULL_RENDER"
echo "    OVERWRITE_SPLITS=$OVERWRITE_SPLITS"

SPLITS_DIR="$SPLITS_DIR" \
FULL_RESULT_BASE="$FULL_RESULT_BASE" \
OUTPUT_RESULT_ROOT="$OUTPUT_RESULT_ROOT" \
RENDER_FOLDER_NAME="$RENDER_FOLDER" \
RESOLUTION="$RESOLUTION" \
NUM_FRAMES="$NUM_FRAMES" \
STATIC_FRAME_STRIDE="$STATIC_FRAME_STRIDE" \
RENDER_CHUNK_SIZE="$RENDER_CHUNK_SIZE" \
DEPTH_OUTLIERS="$DEPTH_OUTLIERS" \
OVERWRITE_FULL_RENDER="$OVERWRITE_FULL_RENDER" \
OVERWRITE_SPLITS="$OVERWRITE_SPLITS" \
bash scripts/test_video/render_shared_static_and_slice.sh "$MANIFEST"

echo
echo "FlowLong shared conditions generated."
echo "Per-window conditions: $OUTPUT_RESULT_ROOT"
