#!/usr/bin/env bash
set -euo pipefail

# for our USP runs, we need to disable NCCL P2P and IB.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$VISTA4D_ROOT"

USE_USP=${USE_USP:-false}
NUM_GPUS=${NUM_GPUS:-8}
TRIAL_INFERENCE_TIME=${TRIAL_INFERENCE_TIME:-false}
SEEDS=${SEEDS:-10027}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}
CFG_SCALE=${CFG_SCALE:-5.0}
SIGMA_SHIFT=${SIGMA_SHIFT:-5.0}

LOCAL_WAN_FOLDER=${LOCAL_WAN_FOLDER:-./checkpoints/wan}
WAN_NAME=${WAN_NAME:-Wan2.1-T2V-14B}
WAN_PATHS="${WAN_NAME}:diffusion_pytorch_model*.safetensors,${WAN_NAME}:models_t5_umt5-xxl-enc-bf16.pth,${WAN_NAME}:Wan2.1_VAE.pth"
TOKENIZER_PATHS="${WAN_NAME}:google/*"

if [[ "$RESOLUTION" == "384p" ]]; then
    VISTA4D_FOLDER=${VISTA4D_FOLDER:-./checkpoints/vista4d/384p49_step=30000}
elif [[ "$RESOLUTION" == "720p" ]]; then
    VISTA4D_FOLDER=${VISTA4D_FOLDER:-./checkpoints/vista4d/720p49_step=3000}
else
    echo "Unrecognized RESOLUTION=$RESOLUTION, exiting script."
    exit 1
fi

INPUT_FOLDER=${INPUT_FOLDER:-$RENDER_OUTPUT_FOLDER}
OUTPUT_FOLDER=${OUTPUT_FOLDER:-$INFERENCE_OUTPUT_FOLDER}

if [[ -z "${PROMPT:-}" ]]; then
    if [[ "$EXAMPLE" == "couple-newspaper" ]]; then
        PROMPT="An elderly couple is seated on a wooden bench in Baleen, deeply engrossed in reading a magazine. The man, wearing a white polo shirt and dark trousers, has sunglasses on his head, while the woman, in a striped top and beige pants, leans on him. They are surrounded by a tranquil suburban setting, featuring a white SUV, a blue sedan, and a U.S. Post Office. As time passes, the scene includes a clear sky, American flags, and a 'BANK OF AMERICA' sign, reflecting a peaceful moment in a quaint town."
    elif [[ "$EXAMPLE" == "couple-walk" ]]; then
        PROMPT="A man in a blue hoodie and green pants, holding a bouquet of yellow flowers, walks alongside a woman in a beige trench coat and white pants, accompanied by a small white dog on a red leash, in front of a red brick building with large windows and black iron bars. The scene transitions to show a white SUV parked on the street, reflecting the bare branches of a tree and the building's facade, suggesting a peaceful urban setting. The couple and their dog continue their walk, passing by the parked SUV, under a clear sky, indicating a pleasant day in an urban environment."
    elif [[ "$EXAMPLE" == "elderly-tennis" ]]; then
        PROMPT="An elderly man in blue and white track pants and sneakers is seen engaging in a game of tennis on an asphalt court, surrounded by autumn leaves. Initially, he is captured mid-stride, casting a long shadow. As time passes, he is shown in various poses, including standing with a tennis racket, preparing to serve, and finally, poised to serve with a red and black racket. His attire includes a light blue jacket and a cap with a green emblem. The setting transitions from a tranquil park to a more urban environment, with the late afternoon sun casting long shadows, indicating a peaceful yet active lifestyle."
    elif [[ "$EXAMPLE" == "mountain-hike" ]]; then
        PROMPT="A solitary hiker, equipped with a backpack and trekking poles, traverses a rocky path in a tranquil, mountainous landscape, surrounded by leafless trees and sparse greenery under a partly cloudy sky. The hiker, dressed in dark attire, stands poised on a boulder, gazing at a majestic, snow-capped mountain peak. The scene, bathed in soft light, suggests it might be early morning or late afternoon. The hiker, now with a green jacket, continues their journey, embodying a sense of peaceful solitude and adventure in the serene wilderness."
    elif [[ "$EXAMPLE" == "park-selfie" ]]; then
        PROMPT="A man with shoulder-length wavy hair, wearing a light blue shirt and teal t-shirt, captures a selfie with a woman in a pink bucket hat and white jacket, seated beside him on a wooden bench in a park. They are surrounded by lush greenery and a calm body of water, under a partly cloudy sky. The man points at the camera, ensuring they are both in the frame, while the woman smiles broadly, her eyes closed in joy. Their relaxed demeanor and the serene park setting reflect a moment of shared happiness and contentment."
    elif [[ "$EXAMPLE" == "parkour" ]]; then
        PROMPT="A video of a man wearing a sky blue shirt and black pants doing parkour, where he jumps over a concrete ledge and then a metal fence. The scene has a mix of concrete paths, plants and greenery, and buildings in the background, and it takes place during a sunny day."
    elif [[ "$EXAMPLE" == "snowboard" ]]; then
        PROMPT="A man wearing a black helmet, snow jacket, and gloves and dark blue snow pants is snowboarding down a path of snow, where the snow looks compact and already has many marks on it from previous snowboarders and skiers. As he snowboards he leaps up with the snowboard and does a 360 jump, flinging clouds of snow around his snowboard. Behind him is a hill or mountain of snow with rocks on it, surrounding the snowboard and skiing path. The scene seems to take place closer to sunset or sunrise, with a long shadow behind the snowboarder."
    elif [[ "$EXAMPLE" == "soapbox" ]]; then
        PROMPT="A video of a man pushing a sky blue homemade cart with a man wearing a helmet and sitting on top of it and a red barrel at the front of the cart. The man runs with and pushes the cart on top of a wooden soapbox with ramps on either side. The scene takes place in the middle of a street with houses and spectators around, during a half-cloudly half-sunny day."
    else
        PROMPT="$DEFAULT_TEST_VIDEO_PROMPT"
    fi
fi

read -r -a SEED <<< "$SEEDS"

ARGS=()
if [[ "${TILE_VAE:-false}" == "true" ]]; then
    ARGS+=(--tile_vae)
fi
if [[ "$USE_USP" == "true" ]]; then
    ARGS+=(--use_usp)
    RUN=(torchrun --standalone --nproc_per_node "$NUM_GPUS")
else
    RUN=(python3)
fi
if [[ "$TRIAL_INFERENCE_TIME" == "true" ]]; then
    ARGS+=(--num_inference_time_trials 3)
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    read -r -a EXTRA_ARGS_ARRAY <<< "$EXTRA_ARGS"
    ARGS+=("${EXTRA_ARGS_ARRAY[@]}")
fi

if [[ ! -d "$INPUT_FOLDER" ]]; then
    echo "Input folder not found: $INPUT_FOLDER"
    echo "Run scripts/test_video/render.sh first, or override INPUT_FOLDER."
    exit 1
fi

echo "Script kwargs:"
echo "    SOURCE_VIDEO=$SOURCE_VIDEO"
echo "    EXAMPLE=$EXAMPLE"
echo "    RESULT_ROOT=$RESULT_ROOT"
echo "    RESOLUTION=$RESOLUTION"
echo "    USE_USP=$USE_USP"
echo "    NUM_GPUS=$NUM_GPUS"
echo "    TRIAL_INFERENCE_TIME=$TRIAL_INFERENCE_TIME"
echo "    RENDER_FOLDER=$RENDER_FOLDER"
echo "    INPUT_FOLDER=$INPUT_FOLDER"
echo "    OUTPUT_FOLDER=$OUTPUT_FOLDER"
echo "    VISTA4D_FOLDER=$VISTA4D_FOLDER"
echo "    HEIGHT=$HEIGHT"
echo "    WIDTH=$WIDTH"
echo "    NUM_FRAMES=$NUM_FRAMES"
echo "    SEEDS=$SEEDS"
echo "    NUM_INFERENCE_STEPS=$NUM_INFERENCE_STEPS"
echo "    CFG_SCALE=$CFG_SCALE"
echo "    SIGMA_SHIFT=$SIGMA_SHIFT"

"${RUN[@]}" -m scripts.inference.inference \
    --model_id_with_origin_paths "$WAN_PATHS" \
    --tokenizer_id_with_origin_path "$TOKENIZER_PATHS" \
    --local_model_folder "$LOCAL_WAN_FOLDER" \
    --vista4d_checkpoint "${VISTA4D_CHECKPOINT:-$VISTA4D_FOLDER}" \
    --vista4d_config_path "$VISTA4D_FOLDER/config.yaml" \
    --input_folder "$INPUT_FOLDER" \
    --output_folder "$OUTPUT_FOLDER" \
    --prompt "$PROMPT" \
    --height "$HEIGHT" --width "$WIDTH" --num_frames "$NUM_FRAMES" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --cfg_scale "$CFG_SCALE" \
    --sigma_shift "$SIGMA_SHIFT" \
    --seed "${SEED[@]}" \
    --save_gif \
    "${ARGS[@]}"
