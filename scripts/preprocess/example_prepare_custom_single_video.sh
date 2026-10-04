INPUT=${INPUT:-../data/PRO_VID_20260602_153852_00_023/PRO_VID_20260602_153852_00_023_ud.mp4}
OUTPUT_NAME=${OUTPUT_NAME:-PRO_VID_20260602_153852_00_023_frames170_218_720p49}
START_FRAME=${START_FRAME:-170}
NUM_FRAMES=${NUM_FRAMES:-49}
HEIGHT=${HEIGHT:-720}
WIDTH=${WIDTH:-1280}
QUALITY=${QUALITY:-9}

echo "Script kwargs:"
echo "    INPUT=$INPUT"
echo "    OUTPUT_NAME=$OUTPUT_NAME"
echo "    START_FRAME=$START_FRAME"
echo "    NUM_FRAMES=$NUM_FRAMES"
echo "    HEIGHT=$HEIGHT"
echo "    WIDTH=$WIDTH"
echo "    QUALITY=$QUALITY"

python3 -m scripts.preprocess.prepare_custom_single_video \
    --input "$INPUT" \
    --output_dir ./media/single \
    --output_name "$OUTPUT_NAME" \
    --start_frame "$START_FRAME" \
    --num_frames "$NUM_FRAMES" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --quality "$QUALITY"
