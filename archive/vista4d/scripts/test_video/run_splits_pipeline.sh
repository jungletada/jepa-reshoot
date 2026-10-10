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
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

TARGET=${1:-${SPLIT_TARGET:-}}
RESOLUTION=${RESOLUTION:-384p}
if [[ -z "${SPLITS_DIR+x}" ]]; then
    if [[ "$RESOLUTION" == "720p" ]]; then
        SPLITS_DIR=./media/splits/720p
    else
        SPLITS_DIR=./media/splits
    fi
fi
STEPS=${STEPS:-"render inference"}
NUM_FRAMES=${NUM_FRAMES:-49}
USE_SMOOTHED_CAMERA=${USE_SMOOTHED_CAMERA:-true}
SMOOTHED_CAMERA_NAME=${SMOOTHED_CAMERA_NAME:-cameras_gaussian_smooth.npz}
DRY_RUN=${DRY_RUN:-false}
FORCE=${FORCE:-false}
START_INDEX=${START_INDEX:-}
END_INDEX=${END_INDEX:-}
STOP_ON_ERROR=${STOP_ON_ERROR:-true}
LOG_ROOT=${LOG_ROOT:-./logs/test_video_splits}

if [[ -z "$TARGET" ]]; then
    echo "Usage: bash scripts/test_video/run_splits_pipeline.sh <source_stem|source_video|manifest.csv>"
    echo
    echo "Examples:"
    echo "  bash scripts/test_video/run_splits_pipeline.sh PRO_VID_20260602_153852_00_023_ud"
    echo "  bash scripts/test_video/run_splits_recon.sh foo"
    echo "  STEPS=\"render inference\" bash scripts/test_video/run_splits_pipeline.sh foo"
    echo "  DRY_RUN=true bash scripts/test_video/run_splits_pipeline.sh ./media/splits/foo_splits_manifest.csv"
    echo
    echo "Optional env vars:"
    echo "  SPLITS_DIR=$SPLITS_DIR"
    echo "  STEPS=\"$STEPS\""
    echo "  RESOLUTION=$RESOLUTION"
    echo "  NUM_FRAMES=$NUM_FRAMES"
    echo "  USE_SMOOTHED_CAMERA=$USE_SMOOTHED_CAMERA"
    echo "  START_INDEX=<empty means first split>"
    echo "  END_INDEX=<empty means last split>"
    echo "  FORCE=$FORCE"
    echo "  DRY_RUN=$DRY_RUN"
    exit 1
fi

for requested_step in $STEPS; do
    if [[ "$requested_step" == "recon" || "$requested_step" == "smooth" ]]; then
        echo "The split pipeline no longer runs recon/smooth independently per clip." >&2
        echo "Run the full-sequence workflow first:" >&2
        echo "  bash scripts/test_video/run_splits_recon.sh $TARGET" >&2
        echo "Then use STEPS=\"render inference\" here." >&2
        exit 1
    fi
done

derive_source_stem() {
    local target="$1"
    local base
    base="$(basename "$target")"
    base="${base%.*}"
    base="${base%_splits_manifest}"
    echo "$base"
}

derive_example_name() {
    local source_video="$1"
    local resolution="$2"
    local num_frames="$3"
    local stem
    stem="$(basename "$source_video")"
    stem="${stem%.*}"
    case "$stem" in
        *_384p"$num_frames")
            echo "${stem%_384p$num_frames}_${resolution}${num_frames}"
            ;;
        *_720p"$num_frames")
            echo "${stem%_720p$num_frames}_${resolution}${num_frames}"
            ;;
        *)
            echo "$stem"
            ;;
    esac
}

render_folder_for_clip() {
    if [[ -n "${RENDER_FOLDER:-}" ]]; then
        echo "$RENDER_FOLDER"
    elif [[ "$USE_SMOOTHED_CAMERA" == "true" ]]; then
        echo "render_${RESOLUTION}_smooth"
    else
        echo "render_$RESOLUTION"
    fi
}

inference_folder_for_clip() {
    local render_folder="$1"
    if [[ "$render_folder" == *_smooth ]]; then
        echo "vista4d_${RESOLUTION}_smooth"
    else
        echo "vista4d_$RESOLUTION"
    fi
}

read_manifest_csv() {
    local manifest="$1"
    tail -n +2 "$manifest" | awk -F',' '{gsub(/\r$/, "", $5); print $5}' | sed '/^[[:space:]]*$/d' | while IFS= read -r clip; do
        if [[ -f "$clip" ]]; then
            echo "$clip"
        else
            echo "$SPLITS_DIR/$(basename "$clip")"
        fi
    done
}

collect_splits() {
    local target="$1"
    local stem

    if [[ -f "$target" && "$target" == *.csv ]]; then
        read_manifest_csv "$target"
        return
    fi

    if [[ -f "$target" && "$target" == *.json ]]; then
        local csv_manifest="${target%.json}.csv"
        if [[ ! -f "$csv_manifest" ]]; then
            echo "JSON manifest was provided, but matching CSV manifest was not found: $csv_manifest" >&2
            exit 1
        fi
        read_manifest_csv "$csv_manifest"
        return
    fi

    stem="$(derive_source_stem "$target")"
    local csv_manifest="$SPLITS_DIR/${stem}_splits_manifest.csv"
    if [[ -f "$csv_manifest" ]]; then
        read_manifest_csv "$csv_manifest"
        return
    fi
    find "$SPLITS_DIR" -maxdepth 1 -type f -name "${stem}_split*.mp4" | sort
}

filter_splits_by_index() {
    local index=0
    while IFS= read -r clip; do
        [[ -z "$clip" ]] && continue
        if [[ -n "$START_INDEX" && "$index" -lt "$START_INDEX" ]]; then
            index=$((index + 1))
            continue
        fi
        if [[ -n "$END_INDEX" && "$index" -gt "$END_INDEX" ]]; then
            break
        fi
        echo "$clip"
        index=$((index + 1))
    done
}

should_skip_step() {
    local step="$1"
    local result_root="$2"
    local render_folder="$3"
    local inference_folder="$4"

    if [[ "$FORCE" == "true" ]]; then
        return 1
    fi

    case "$step" in
        render)
            local camera_condition="$result_root/recon_and_seg/cameras.npz"
            if [[ "$USE_SMOOTHED_CAMERA" == "true" ]]; then
                camera_condition="$result_root/recon_and_seg/$SMOOTHED_CAMERA_NAME"
            fi
            [[ -f "$camera_condition" \
                && -f "$result_root/$render_folder/video_src.mp4" \
                && -f "$result_root/$render_folder/video_pc.mp4" \
                && "$result_root/$render_folder/video_src.mp4" -nt "$camera_condition" \
                && "$result_root/$render_folder/video_pc.mp4" -nt "$camera_condition" ]]
            ;;
        inference)
            local render_condition="$result_root/$render_folder/video_pc.mp4"
            [[ -f "$render_condition" ]] || return 1
            local generated
            for generated in "$result_root/$inference_folder"/video_seed=*.mp4; do
                if [[ -f "$generated" && "$generated" -nt "$render_condition" ]]; then
                    return 0
                fi
            done
            return 1
            ;;
        *)
            return 1
            ;;
    esac
}

step_script() {
    case "$1" in
        render) echo "scripts/test_video/render.sh" ;;
        inference) echo "scripts/test_video/inference.sh" ;;
        *)
            echo "Unknown step: $1" >&2
            exit 1
            ;;
    esac
}

run_step() {
    local step="$1"
    local clip="$2"
    local clip_index="$3"
    local example="$4"
    local result_root="$5"
    local log_dir="$6"
    local script
    script="$(step_script "$step")"

    local log_file="$log_dir/${clip_index}_${example}_${step}.log"
    echo
    echo "[$clip_index][$step] SOURCE_VIDEO=$clip"
    echo "[$clip_index][$step] LOG=$log_file"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY_RUN: SOURCE_VIDEO=\"$clip\" RESOLUTION=\"$RESOLUTION\" NUM_FRAMES=\"$NUM_FRAMES\" bash $script"
        return 0
    fi

    (
        unset EXAMPLE RESULT_ROOT RECON_AND_SEG_FOLDER SOURCE_CAMERA_PATH SMOOTHED_CAMERA_PATH
        unset CAM_PATH RENDER_OUTPUT_FOLDER INFERENCE_OUTPUT_FOLDER INPUT_FOLDER OUTPUT_FOLDER VIDEO_PATH
        export SOURCE_VIDEO="$clip"
        export RESOLUTION="$RESOLUTION"
        export NUM_FRAMES="$NUM_FRAMES"
        export USE_SMOOTHED_CAMERA="$USE_SMOOTHED_CAMERA"
        export SMOOTHED_CAMERA_NAME="$SMOOTHED_CAMERA_NAME"
        bash "$script"
    ) 2>&1 | tee "$log_file"
}

mapfile -t SPLIT_CLIPS < <(collect_splits "$TARGET" | filter_splits_by_index)
if [[ "${#SPLIT_CLIPS[@]}" -eq 0 ]]; then
    echo "No split clips found for target: $TARGET"
    echo "Searched under: $SPLITS_DIR"
    exit 1
fi

SOURCE_STEM="$(derive_source_stem "$TARGET")"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$LOG_ROOT/${SOURCE_STEM}_${RESOLUTION}_${RUN_ID}"
mkdir -p "$LOG_DIR"

echo "Batch kwargs:"
echo "    TARGET=$TARGET"
echo "    SOURCE_STEM=$SOURCE_STEM"
echo "    SPLITS_DIR=$SPLITS_DIR"
echo "    NUM_SPLITS=${#SPLIT_CLIPS[@]}"
echo "    STEPS=$STEPS"
echo "    RESOLUTION=$RESOLUTION"
echo "    NUM_FRAMES=$NUM_FRAMES"
echo "    USE_SMOOTHED_CAMERA=$USE_SMOOTHED_CAMERA"
echo "    FORCE=$FORCE"
echo "    DRY_RUN=$DRY_RUN"
echo "    LOG_DIR=$LOG_DIR"

printf "%s\n" "${SPLIT_CLIPS[@]}" > "$LOG_DIR/split_clips.txt"

for clip_i in "${!SPLIT_CLIPS[@]}"; do
    clip="${SPLIT_CLIPS[$clip_i]}"
    example="$(derive_example_name "$clip" "$RESOLUTION" "$NUM_FRAMES")"
    result_root="./results/single/$example"
    render_folder="$(render_folder_for_clip)"
    inference_folder="$(inference_folder_for_clip "$render_folder")"

    echo
    echo "==== Split $clip_i / $((${#SPLIT_CLIPS[@]} - 1)): $clip ===="
    echo "     EXAMPLE=$example"
    echo "     RESULT_ROOT=$result_root"
    echo "     RENDER_FOLDER=$render_folder"
    echo "     INFERENCE_FOLDER=$inference_folder"

    for step in $STEPS; do
        if should_skip_step "$step" "$result_root" "$render_folder" "$inference_folder"; then
            echo "[$clip_i][$step] skip existing output"
            continue
        fi

        if ! run_step "$step" "$clip" "$clip_i" "$example" "$result_root" "$LOG_DIR"; then
            echo "[$clip_i][$step] failed"
            if [[ "$STOP_ON_ERROR" == "true" ]]; then
                exit 1
            fi
        fi
    done
done

echo
echo "Batch pipeline finished."
echo "Logs: $LOG_DIR"
