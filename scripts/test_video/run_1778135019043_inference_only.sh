#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISTA4D_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$VISTA4D_ROOT"

export RESOLUTION=${RESOLUTION:-384p}
export NUM_FRAMES=${NUM_FRAMES:-49}
export USE_SMOOTHED_CAMERA=${USE_SMOOTHED_CAMERA:-true}
export RENDER_FOLDER=${RENDER_FOLDER:-render_384p_smooth}
export STEPS="inference"

export USE_USP=${USE_USP:-false}
export NUM_GPUS=${NUM_GPUS:-8}
export SEEDS=${SEEDS:-10027}
export PROMPT=${PROMPT:-"A realistic wide-angle smartphone video on a sunny day, natural body motion, bright daylight, realistic shadows, and detailed outdoor surroundings."}

TARGET=${TARGET:-1778135019043}

echo "Running Vista4D inference-only pipeline:"
echo "    TARGET=$TARGET"
echo "    RESOLUTION=$RESOLUTION"
echo "    RENDER_FOLDER=$RENDER_FOLDER"
echo "    USE_USP=$USE_USP"
echo "    NUM_GPUS=$NUM_GPUS"
echo "    SEEDS=$SEEDS"
echo "    START_INDEX=${START_INDEX:-<first>}"
echo "    END_INDEX=${END_INDEX:-<last>}"
echo "    EXTRA_ARGS=${EXTRA_ARGS:-<none>}"

bash scripts/test_video/run_splits_pipeline.sh "$TARGET"
