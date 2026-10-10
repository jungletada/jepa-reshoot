#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISTA4D_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$VISTA4D_ROOT"

# AUTO_ACTIVATE_CONDA=${AUTO_ACTIVATE_CONDA:-true}
# if [[ "$AUTO_ACTIVATE_CONDA" == "true" && "${CONDA_DEFAULT_ENV:-}" != "vista4d" ]]; then
#     set +u
#     source /home/peng/anaconda3/etc/profile.d/conda.sh
#     conda activate vista4d
#     set -u
# fi

INPUT=${1:-${INPUT:-}}
RESOLUTION=${RESOLUTION:-384p}
if [[ -z "${OUTPUT_DIR+x}" ]]; then
    if [[ "$RESOLUTION" == "720p" ]]; then
        OUTPUT_DIR=./media/splits/720p
    else
        OUTPUT_DIR=./media/splits
    fi
fi
CLIP_FRAMES=${CLIP_FRAMES:-49}
OVERLAP=${OVERLAP:-5}
TEMPORAL_ALIGNMENT=${TEMPORAL_ALIGNMENT:-4}
START_FRAME=${START_FRAME:-0}
MAX_FRAMES=${MAX_FRAMES:-}
if [[ "$RESOLUTION" == "384p" ]]; then
    HEIGHT=${HEIGHT:-384}
    WIDTH=${WIDTH:-672}
elif [[ "$RESOLUTION" == "720p" ]]; then
    HEIGHT=${HEIGHT:-720}
    WIDTH=${WIDTH:-1280}
else
    echo "Unrecognized RESOLUTION=$RESOLUTION, expected 384p or 720p."
    exit 1
fi
QUALITY=${QUALITY:-9}
INCLUDE_TAIL=${INCLUDE_TAIL:-true}

if [[ -z "$INPUT" ]]; then
    echo "Usage: bash scripts/test_video/split_video.sh <input_video>"
    echo
    echo "Optional env vars:"
    echo "  OUTPUT_DIR=$OUTPUT_DIR"
    echo "  CLIP_FRAMES=$CLIP_FRAMES"
    echo "  OVERLAP=$OVERLAP"
    echo "  TEMPORAL_ALIGNMENT=$TEMPORAL_ALIGNMENT"
    echo "  START_FRAME=$START_FRAME"
    echo "  MAX_FRAMES=<empty means full video>"
    echo "  RESOLUTION=$RESOLUTION"
    echo "  HEIGHT=$HEIGHT"
    echo "  WIDTH=$WIDTH"
    echo "  QUALITY=$QUALITY"
    echo "  INCLUDE_TAIL=$INCLUDE_TAIL"
    exit 1
fi

ARGS=()
if [[ -n "$MAX_FRAMES" ]]; then
    ARGS+=(--max_frames "$MAX_FRAMES")
fi
if [[ "$INCLUDE_TAIL" != "true" ]]; then
    ARGS+=(--no_include_tail)
fi

echo "Script kwargs:"
echo "    INPUT=$INPUT"
echo "    OUTPUT_DIR=$OUTPUT_DIR"
echo "    CLIP_FRAMES=$CLIP_FRAMES"
echo "    OVERLAP=$OVERLAP"
echo "    TEMPORAL_ALIGNMENT=$TEMPORAL_ALIGNMENT"
echo "    START_FRAME=$START_FRAME"
echo "    MAX_FRAMES=${MAX_FRAMES:-<full video>}"
echo "    RESOLUTION=$RESOLUTION"
echo "    HEIGHT=$HEIGHT"
echo "    WIDTH=$WIDTH"
echo "    QUALITY=$QUALITY"
echo "    INCLUDE_TAIL=$INCLUDE_TAIL"

python3 -m scripts.preprocess.split_video_into_clips \
    --input "$INPUT" \
    --output_dir "$OUTPUT_DIR" \
    --clip_frames "$CLIP_FRAMES" \
    --overlap "$OVERLAP" \
    --temporal_alignment "$TEMPORAL_ALIGNMENT" \
    --start_frame "$START_FRAME" \
    --resolution "$RESOLUTION" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --quality "$QUALITY" \
    "${ARGS[@]}"
