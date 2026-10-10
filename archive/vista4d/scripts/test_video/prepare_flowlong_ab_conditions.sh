#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISTA4D_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$VISTA4D_ROOT"

TARGET=${1:-${SPLIT_TARGET:-}}
RESOLUTION=${RESOLUTION:-384p}
NUM_FRAMES=${NUM_FRAMES:-49}
BASELINE_CONDITION_ROOT=${BASELINE_CONDITION_ROOT:-./results/shared_static_single}
FLOWLONG_RESULT_ROOT=${FLOWLONG_RESULT_ROOT:-./results/flowlong_single}
FULL_RESULT_BASE=${FULL_RESULT_BASE:-./results/full}
STATIC_FRAME_STRIDE=${STATIC_FRAME_STRIDE:-4}
RENDER_CHUNK_SIZE=${RENDER_CHUNK_SIZE:-4}
DEPTH_OUTLIERS=${DEPTH_OUTLIERS:-gaussian}
OVERWRITE_FULL_RENDER=${OVERWRITE_FULL_RENDER:-false}
OVERWRITE_BASELINE_SPLITS=${OVERWRITE_BASELINE_SPLITS:-false}
OVERWRITE_FLOWLONG_SPLITS=${OVERWRITE_FLOWLONG_SPLITS:-false}

if [[ -z "$TARGET" ]]; then
    echo "Usage: bash scripts/test_video/prepare_flowlong_ab_conditions.sh <source_stem>" >&2
    exit 1
fi

case "$RESOLUTION" in
    384p)
        EXPECTED_HEIGHT=384
        EXPECTED_WIDTH=672
        DEFAULT_BASELINE_SPLITS_DIR=./media/splits
        DEFAULT_FLOWLONG_SPLITS_DIR=./media/flowlong_splits
        ;;
    720p)
        EXPECTED_HEIGHT=720
        EXPECTED_WIDTH=1280
        DEFAULT_BASELINE_SPLITS_DIR=./media/splits/720p
        DEFAULT_FLOWLONG_SPLITS_DIR=./media/flowlong_splits/720p
        ;;
    *)
        echo "Unrecognized RESOLUTION=$RESOLUTION, expected 384p or 720p." >&2
        exit 2
        ;;
esac
HEIGHT=${HEIGHT:-$EXPECTED_HEIGHT}
WIDTH=${WIDTH:-$EXPECTED_WIDTH}
BASELINE_SPLITS_DIR=${BASELINE_SPLITS_DIR:-$DEFAULT_BASELINE_SPLITS_DIR}
FLOWLONG_SPLITS_DIR=${FLOWLONG_SPLITS_DIR:-$DEFAULT_FLOWLONG_SPLITS_DIR}
if [[ "$HEIGHT" != "$EXPECTED_HEIGHT" || "$WIDTH" != "$EXPECTED_WIDTH" ]]; then
    echo "RESOLUTION=$RESOLUTION requires WIDTH x HEIGHT " \
        "$EXPECTED_WIDTH x $EXPECTED_HEIGHT, got $WIDTH x $HEIGHT." >&2
    exit 2
fi

SOURCE_STEM="$(basename "$TARGET")"
SOURCE_STEM="${SOURCE_STEM%.*}"
SOURCE_STEM="${SOURCE_STEM%_splits_manifest}"
BASELINE_MANIFEST="$BASELINE_SPLITS_DIR/${SOURCE_STEM}_splits_manifest.json"
FLOWLONG_MANIFEST="$FLOWLONG_SPLITS_DIR/${SOURCE_STEM}_splits_manifest.json"
FULL_SEQUENCE_ROOT=${FULL_SEQUENCE_ROOT:-$FULL_RESULT_BASE/${SOURCE_STEM}_stitched_${RESOLUTION}}
FULL_RECON_FOLDER=${FULL_RECON_FOLDER:-$FULL_SEQUENCE_ROOT/recon_and_seg}
FULL_RENDER_FOLDER=${FULL_RENDER_FOLDER:-$FULL_SEQUENCE_ROOT/render_${RESOLUTION}_smooth_shared_static}
RENDER_FOLDER=${RENDER_FOLDER:-render_${RESOLUTION}_smooth}

for path in "$BASELINE_MANIFEST" "$FLOWLONG_MANIFEST"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing manifest: $path" >&2
        exit 1
    fi
done
if [[ ! -d "$FULL_RECON_FOLDER" ]]; then
    echo "Missing full reconstruction: $FULL_RECON_FOLDER" >&2
    echo "Run the full-sequence reconstruction step documented in READMEv2.md first." >&2
    exit 1
fi

python3 - \
    "$BASELINE_MANIFEST" \
    "$FLOWLONG_MANIFEST" \
    "$RESOLUTION" \
    "$HEIGHT" \
    "$WIDTH" <<'PY'
import json
from pathlib import Path
import sys

from utils.resolution import validate_manifest_resolution
from utils.split_manifest import validate_flowlong_window_manifest

baseline_path = Path(sys.argv[1])
flowlong_path = Path(sys.argv[2])
resolution = sys.argv[3]
height = int(sys.argv[4])
width = int(sys.argv[5])
baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
flowlong = json.loads(flowlong_path.read_text(encoding="utf-8"))

for manifest in (baseline, flowlong):
    validate_manifest_resolution(
        manifest,
        resolution=resolution,
        height=height,
        width=width,
    )

validate_flowlong_window_manifest(
    baseline,
    clip_frames=49,
    overlap=5,
    temporal_alignment=4,
)
validate_flowlong_window_manifest(
    flowlong,
    clip_frames=49,
    overlap=25,
    temporal_alignment=4,
)

for key in ("input_path", "start_frame", "end_exclusive", "fps"):
    if baseline.get(key) != flowlong.get(key):
        raise ValueError(
            f"Baseline and FlowLong manifests differ at {key}: "
            f"{baseline.get(key)!r} != {flowlong.get(key)!r}"
        )
print("Window manifest pair check: PASS")
PY

echo "FlowLong A/B shared-condition kwargs:"
echo "    TARGET=$SOURCE_STEM"
echo "    RESOLUTION=$RESOLUTION ($WIDTH x $HEIGHT)"
echo "    BASELINE_MANIFEST=$BASELINE_MANIFEST"
echo "    FLOWLONG_MANIFEST=$FLOWLONG_MANIFEST"
echo "    FULL_RECON_FOLDER=$FULL_RECON_FOLDER"
echo "    FULL_RENDER_FOLDER=$FULL_RENDER_FOLDER"
echo "    BASELINE_CONDITION_ROOT=$BASELINE_CONDITION_ROOT"
echo "    FLOWLONG_RESULT_ROOT=$FLOWLONG_RESULT_ROOT"

SPLITS_DIR="$BASELINE_SPLITS_DIR" \
FULL_RESULT_BASE="$FULL_RESULT_BASE" \
FULL_SEQUENCE_ROOT="$FULL_SEQUENCE_ROOT" \
FULL_RECON_FOLDER="$FULL_RECON_FOLDER" \
FULL_RENDER_FOLDER="$FULL_RENDER_FOLDER" \
OUTPUT_RESULT_ROOT="$BASELINE_CONDITION_ROOT" \
RENDER_FOLDER_NAME="$RENDER_FOLDER" \
RESOLUTION="$RESOLUTION" \
HEIGHT="$HEIGHT" \
WIDTH="$WIDTH" \
NUM_FRAMES="$NUM_FRAMES" \
STATIC_FRAME_STRIDE="$STATIC_FRAME_STRIDE" \
RENDER_CHUNK_SIZE="$RENDER_CHUNK_SIZE" \
DEPTH_OUTLIERS="$DEPTH_OUTLIERS" \
OVERWRITE_FULL_RENDER="$OVERWRITE_FULL_RENDER" \
OVERWRITE_SPLITS="$OVERWRITE_BASELINE_SPLITS" \
bash scripts/test_video/render_shared_static_and_slice.sh "$BASELINE_MANIFEST"

SPLITS_DIR="$FLOWLONG_SPLITS_DIR" \
FULL_RESULT_BASE="$FULL_RESULT_BASE" \
FULL_SEQUENCE_ROOT="$FULL_SEQUENCE_ROOT" \
FULL_RECON_FOLDER="$FULL_RECON_FOLDER" \
FULL_RENDER_FOLDER="$FULL_RENDER_FOLDER" \
OUTPUT_RESULT_ROOT="$FLOWLONG_RESULT_ROOT" \
RENDER_FOLDER="$RENDER_FOLDER" \
RESOLUTION="$RESOLUTION" \
HEIGHT="$HEIGHT" \
WIDTH="$WIDTH" \
NUM_FRAMES="$NUM_FRAMES" \
STATIC_FRAME_STRIDE="$STATIC_FRAME_STRIDE" \
RENDER_CHUNK_SIZE="$RENDER_CHUNK_SIZE" \
DEPTH_OUTLIERS="$DEPTH_OUTLIERS" \
OVERWRITE_FULL_RENDER=false \
OVERWRITE_SPLITS="$OVERWRITE_FLOWLONG_SPLITS" \
bash scripts/test_video/prepare_flowlong_conditions.sh "$FLOWLONG_MANIFEST"

echo
echo "Baseline / FlowLong shared conditions are ready."
echo "Shared full render: $FULL_RENDER_FOLDER"
