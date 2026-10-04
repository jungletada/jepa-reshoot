#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISTA4D_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$VISTA4D_ROOT"

TARGET=${1:-${SPLIT_TARGET:-}}
RESOLUTION=${RESOLUTION:-384p}
if [[ -z "${FLOWLONG_SPLITS_DIR+x}" ]]; then
    if [[ "$RESOLUTION" == "720p" ]]; then
        FLOWLONG_SPLITS_DIR=./media/flowlong_splits/720p
    else
        FLOWLONG_SPLITS_DIR=./media/flowlong_splits
    fi
fi
FLOWLONG_RESULT_ROOT=${FLOWLONG_RESULT_ROOT:-./results/flowlong_single}
FLOWLONG_OUTPUT_ROOT=${FLOWLONG_OUTPUT_ROOT:-./results/flowlong}
NUM_FRAMES=${NUM_FRAMES:-49}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}
SIGMA_SHIFT=${SIGMA_SHIFT:-5.0}
CFG_SCALE=${CFG_SCALE:-5.0}
FLOWLONG_STOCHASTIC_THRESHOLD=${FLOWLONG_STOCHASTIC_THRESHOLD:-0.6}
FLOWLONG_MICROBATCH_SIZE=${FLOWLONG_MICROBATCH_SIZE:-1}
FLOWLONG_DISABLE_STOCHASTIC=${FLOWLONG_DISABLE_STOCHASTIC:-false}
SEEDS=${SEEDS:-10027}
FORCE=${FORCE:-false}
USE_USP=${USE_USP:-false}
CFG_MERGE=${CFG_MERGE:-false}
TILE_VAE=${TILE_VAE:-true}

LOCAL_WAN_FOLDER=${LOCAL_WAN_FOLDER:-./checkpoints/wan}
WAN_NAME=${WAN_NAME:-Wan2.1-T2V-14B}
WAN_PATHS="${WAN_NAME}:diffusion_pytorch_model*.safetensors,${WAN_NAME}:models_t5_umt5-xxl-enc-bf16.pth,${WAN_NAME}:Wan2.1_VAE.pth"
TOKENIZER_PATHS="${WAN_NAME}:google/*"

if [[ -z "$TARGET" ]]; then
    echo "Usage: bash scripts/test_video/run_flowlong_inference.sh <source_stem|manifest.json>"
    exit 1
fi
if [[ "$USE_USP" != "false" ]]; then
    echo "FlowLong v1 requires USE_USP=false." >&2
    exit 2
fi
if [[ "$CFG_MERGE" != "false" ]]; then
    echo "FlowLong v1 requires CFG_MERGE=false." >&2
    exit 2
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
    local manifest="$FLOWLONG_SPLITS_DIR/${base}_splits_manifest.json"
    if [[ ! -f "$manifest" ]]; then
        echo "Could not find FlowLong JSON manifest: $manifest" >&2
        exit 1
    fi
    echo "$manifest"
}

MANIFEST="$(resolve_manifest "$TARGET")"
SOURCE_STEM="$(basename "$MANIFEST")"
SOURCE_STEM="${SOURCE_STEM%_splits_manifest.json}"
RENDER_FOLDER=${RENDER_FOLDER:-render_${RESOLUTION}_smooth}
OUTPUT_FOLDER=${OUTPUT_FOLDER:-$FLOWLONG_OUTPUT_ROOT/${SOURCE_STEM}_vista4d_${RESOLUTION}_smooth}

if [[ "$RESOLUTION" == "384p" ]]; then
    EXPECTED_HEIGHT=384
    EXPECTED_WIDTH=672
    VISTA4D_FOLDER=${VISTA4D_FOLDER:-./checkpoints/vista4d/384p49_step=30000}
elif [[ "$RESOLUTION" == "720p" ]]; then
    EXPECTED_HEIGHT=720
    EXPECTED_WIDTH=1280
    VISTA4D_FOLDER=${VISTA4D_FOLDER:-./checkpoints/vista4d/720p49_step=3000}
else
    echo "Unrecognized RESOLUTION=$RESOLUTION, expected 384p or 720p." >&2
    exit 2
fi
HEIGHT=${HEIGHT:-$EXPECTED_HEIGHT}
WIDTH=${WIDTH:-$EXPECTED_WIDTH}
if [[ "$HEIGHT" != "$EXPECTED_HEIGHT" || "$WIDTH" != "$EXPECTED_WIDTH" ]]; then
    echo "RESOLUTION=$RESOLUTION requires WIDTH x HEIGHT " \
        "$EXPECTED_WIDTH x $EXPECTED_HEIGHT, got $WIDTH x $HEIGHT." >&2
    exit 2
fi
VISTA4D_CHECKPOINT=$(python3 -m utils.vista4d_checkpoint resolve "${VISTA4D_CHECKPOINT:-$VISTA4D_FOLDER}")
if [[ ! -f "$VISTA4D_FOLDER/config.yaml" ]]; then
    echo "Incomplete Vista4D checkpoint folder: $VISTA4D_FOLDER" >&2
    exit 1
fi

PROMPT=${PROMPT:-"A realistic handheld smartphone video of people in an everyday scene, with natural body motion, realistic lighting, stable camera motion, and detailed surroundings."}
VISTA4D_CHECKPOINT_SHA256=${VISTA4D_CHECKPOINT_SHA256:-$(python3 -m utils.vista4d_checkpoint sha256 "$VISTA4D_CHECKPOINT")}
read -r -a REQUESTED_SEEDS <<< "$SEEDS"
PENDING_SEEDS=()
for seed in "${REQUESTED_SEEDS[@]}"; do
    video_path="$OUTPUT_FOLDER/video_seed=${seed}.mp4"
    report_path="$OUTPUT_FOLDER/flowlong_report_seed=${seed}.json"
    if [[ "$FORCE" != "true" && -f "$video_path" && -f "$report_path" ]]; then
        python3 - "$report_path" "$VISTA4D_CHECKPOINT_SHA256" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
if report.get('model', {}).get('vista4d_checkpoint', {}).get('sha256') != sys.argv[2]:
    raise SystemExit('Existing output uses a different/unknown checkpoint; use a new OUTPUT_FOLDER or explicitly overwrite')
PY
        echo "Skip completed FlowLong seed=$seed: $video_path"
    else
        PENDING_SEEDS+=("$seed")
    fi
done
if [[ ${#PENDING_SEEDS[@]} -eq 0 ]]; then
    echo "All requested FlowLong seeds are already complete."
    exit 0
fi

echo "FlowLong inference kwargs:"
echo "    MANIFEST=$MANIFEST"
echo "    FLOWLONG_RESULT_ROOT=$FLOWLONG_RESULT_ROOT"
echo "    RENDER_FOLDER=$RENDER_FOLDER"
echo "    OUTPUT_FOLDER=$OUTPUT_FOLDER"
echo "    RESOLUTION=$RESOLUTION ($WIDTH x $HEIGHT)"
echo "    NUM_FRAMES=$NUM_FRAMES"
echo "    USE_USP=$USE_USP"
echo "    CFG_MERGE=$CFG_MERGE"
echo "    NUM_INFERENCE_STEPS=$NUM_INFERENCE_STEPS"
echo "    SIGMA_SHIFT=$SIGMA_SHIFT"
echo "    CFG_SCALE=$CFG_SCALE"
echo "    FLOWLONG_STOCHASTIC_THRESHOLD=$FLOWLONG_STOCHASTIC_THRESHOLD"
echo "    FLOWLONG_MICROBATCH_SIZE=$FLOWLONG_MICROBATCH_SIZE"
echo "    FLOWLONG_DISABLE_STOCHASTIC=$FLOWLONG_DISABLE_STOCHASTIC"
echo "    TILE_VAE=$TILE_VAE"
echo "    PENDING_SEEDS=${PENDING_SEEDS[*]}"
echo "    FORCE=$FORCE"
echo "    LOCAL_WAN_FOLDER=$LOCAL_WAN_FOLDER"
echo "    VISTA4D_FOLDER=$VISTA4D_FOLDER"

ARGS=()
if [[ "$FORCE" == "true" ]]; then
    ARGS+=(--overwrite)
fi
if [[ "$FLOWLONG_DISABLE_STOCHASTIC" == "true" ]]; then
    ARGS+=(--flowlong_disable_stochastic)
fi
if [[ "$TILE_VAE" == "true" ]]; then
    ARGS+=(--tile_vae)
fi
if [[ -n "${NEGATIVE_PROMPT:-}" ]]; then
    ARGS+=(--negative_prompt "$NEGATIVE_PROMPT")
fi
if [[ -n "${VRAM_LIMIT:-}" ]]; then
    ARGS+=(--vram_limit "$VRAM_LIMIT")
fi
if [[ -n "${VISTA4D_CHECKPOINT_SHA256:-}" ]]; then
    ARGS+=(--vista4d_checkpoint_sha256 "$VISTA4D_CHECKPOINT_SHA256")
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
    read -r -a EXTRA_ARGS_ARRAY <<< "$EXTRA_ARGS"
    ARGS+=("${EXTRA_ARGS_ARRAY[@]}")
fi

python3 -m scripts.inference.inference_flowlong \
    --manifest "$MANIFEST" \
    --result_root "$FLOWLONG_RESULT_ROOT" \
    --render_folder "$RENDER_FOLDER" \
    --output_folder "$OUTPUT_FOLDER" \
    --resolution "$RESOLUTION" \
    --model_id_with_origin_paths "$WAN_PATHS" \
    --tokenizer_id_with_origin_path "$TOKENIZER_PATHS" \
    --local_model_folder "$LOCAL_WAN_FOLDER" \
    --vista4d_checkpoint "$VISTA4D_CHECKPOINT" \
    --vista4d_config_path "$VISTA4D_FOLDER/config.yaml" \
    --prompt "$PROMPT" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --num_frames "$NUM_FRAMES" \
    --seed "${PENDING_SEEDS[@]}" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --sigma_shift "$SIGMA_SHIFT" \
    --cfg_scale "$CFG_SCALE" \
    --flowlong_stochastic_threshold "$FLOWLONG_STOCHASTIC_THRESHOLD" \
    --flowlong_microbatch_size "$FLOWLONG_MICROBATCH_SIZE" \
    "${ARGS[@]}"

echo "FlowLong inference finished: $OUTPUT_FOLDER"
