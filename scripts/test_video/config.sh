#!/usr/bin/env bash

TEST_VIDEO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISTA4D_ROOT="$(cd "$TEST_VIDEO_DIR/../.." && pwd)"

SOURCE_VIDEO=${SOURCE_VIDEO:-./media/single/PRO_VID_20260602_153852_00_023_frames300_348_720p49.mp4}
RESOLUTION=${RESOLUTION:-384p}  # 384p, 720p
NUM_FRAMES=${NUM_FRAMES:-49}

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

EXAMPLE=${EXAMPLE:-$(derive_example_name "$SOURCE_VIDEO" "$RESOLUTION" "$NUM_FRAMES")}
RESULT_ROOT=${RESULT_ROOT:-./results/single/$EXAMPLE}
RECON_AND_SEG_FOLDER=${RECON_AND_SEG_FOLDER:-$RESULT_ROOT/recon_and_seg}

SMOOTHED_CAMERA_NAME=${SMOOTHED_CAMERA_NAME:-cameras_gaussian_smooth.npz}
SOURCE_CAMERA_PATH=${SOURCE_CAMERA_PATH:-$RECON_AND_SEG_FOLDER/cameras.npz}
SMOOTHED_CAMERA_PATH=${SMOOTHED_CAMERA_PATH:-$RECON_AND_SEG_FOLDER/$SMOOTHED_CAMERA_NAME}

USE_SMOOTHED_CAMERA=${USE_SMOOTHED_CAMERA:-true}
if [[ -z "${CAM_PATH:-}" ]]; then
    if [[ "$USE_SMOOTHED_CAMERA" == "true" ]]; then
        CAM_PATH="$SMOOTHED_CAMERA_PATH"
    else
        CAM_PATH="$SOURCE_CAMERA_PATH"
    fi
fi

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

if [[ -z "${RENDER_FOLDER:-}" ]]; then
    if [[ "$USE_SMOOTHED_CAMERA" == "true" ]]; then
        RENDER_FOLDER="render_${RESOLUTION}_smooth"
    else
        RENDER_FOLDER="render_$RESOLUTION"
    fi
fi
RENDER_OUTPUT_FOLDER=${RENDER_OUTPUT_FOLDER:-$RESULT_ROOT/$RENDER_FOLDER}

if [[ -z "${INFERENCE_OUTPUT_FOLDER:-}" ]]; then
    if [[ "$RENDER_FOLDER" == *_smooth ]]; then
        INFERENCE_OUTPUT_FOLDER="$RESULT_ROOT/vista4d_${RESOLUTION}_smooth"
    else
        INFERENCE_OUTPUT_FOLDER="$RESULT_ROOT/vista4d_$RESOLUTION"
    fi
fi

DEFAULT_TEST_VIDEO_PROMPT="A realistic handheld smartphone video of people in an everyday scene, with natural body motion, realistic lighting, stable camera motion, and detailed surroundings."
