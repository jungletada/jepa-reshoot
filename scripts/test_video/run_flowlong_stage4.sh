#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISTA4D_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$VISTA4D_ROOT"

TARGET=${1:-${SPLIT_TARGET:-}}
PHASES=${PHASES:-"baseline matching_only t0.5 t0.6 t0.7"}
EVAL_ROOT_BASE=${EVAL_ROOT_BASE:-./results/flowlong_eval}
RESOLUTION=${RESOLUTION:-384p}
if [[ -z "${BASELINE_SPLITS_DIR+x}" ]]; then
    if [[ "$RESOLUTION" == "720p" ]]; then
        BASELINE_SPLITS_DIR=./media/splits/720p
    else
        BASELINE_SPLITS_DIR=./media/splits
    fi
fi
if [[ -z "${FLOWLONG_SPLITS_DIR+x}" ]]; then
    if [[ "$RESOLUTION" == "720p" ]]; then
        FLOWLONG_SPLITS_DIR=./media/flowlong_splits/720p
    else
        FLOWLONG_SPLITS_DIR=./media/flowlong_splits
    fi
fi
BASELINE_CONDITION_ROOT=${BASELINE_CONDITION_ROOT:-./results/shared_static_single}
BASELINE_EXISTING_CLIP_ROOT=${BASELINE_EXISTING_CLIP_ROOT:-}
BASELINE_EXISTING_LOG_DIR=${BASELINE_EXISTING_LOG_DIR:-}
FLOWLONG_RESULT_ROOT=${FLOWLONG_RESULT_ROOT:-./results/flowlong_single}
NUM_FRAMES=${NUM_FRAMES:-49}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}
SIGMA_SHIFT=${SIGMA_SHIFT:-5.0}
CFG_SCALE=${CFG_SCALE:-5.0}
FLOWLONG_MICROBATCH_SIZE=${FLOWLONG_MICROBATCH_SIZE:-1}
SEED=${SEED:-10027}
TILE_VAE=${TILE_VAE:-true}
FORCE=${FORCE:-false}
DRY_RUN=${DRY_RUN:-false}
LOG_ROOT=${LOG_ROOT:-./logs/flowlong_stage4}

LOCAL_WAN_FOLDER=${LOCAL_WAN_FOLDER:-./checkpoints/wan}
WAN_NAME=${WAN_NAME:-Wan2.1-T2V-14B}
WAN_PATHS="${WAN_NAME}:diffusion_pytorch_model*.safetensors,${WAN_NAME}:models_t5_umt5-xxl-enc-bf16.pth,${WAN_NAME}:Wan2.1_VAE.pth"
TOKENIZER_PATHS="${WAN_NAME}:google/*"

if [[ -z "$TARGET" ]]; then
    echo "Usage: bash scripts/test_video/run_flowlong_stage4.sh <source_stem>" >&2
    exit 1
fi
case "$RESOLUTION" in
    384p)
        EXPECTED_HEIGHT=384
        EXPECTED_WIDTH=672
        DEFAULT_VISTA4D_FOLDER=./checkpoints/vista4d/384p49_step=30000
        ;;
    720p)
        EXPECTED_HEIGHT=720
        EXPECTED_WIDTH=1280
        DEFAULT_VISTA4D_FOLDER=./checkpoints/vista4d/720p49_step=3000
        ;;
    *)
        echo "Unrecognized RESOLUTION=$RESOLUTION, expected 384p or 720p." >&2
        exit 2
        ;;
esac
HEIGHT=${HEIGHT:-$EXPECTED_HEIGHT}
WIDTH=${WIDTH:-$EXPECTED_WIDTH}
VISTA4D_FOLDER=${VISTA4D_FOLDER:-$DEFAULT_VISTA4D_FOLDER}
if [[ "$HEIGHT" != "$EXPECTED_HEIGHT" || "$WIDTH" != "$EXPECTED_WIDTH" ]]; then
    echo "RESOLUTION=$RESOLUTION requires WIDTH x HEIGHT " \
        "$EXPECTED_WIDTH x $EXPECTED_HEIGHT, got $WIDTH x $HEIGHT." >&2
    exit 2
fi
if [[ "$NUM_FRAMES" != "49" ]]; then
    echo "Stage 4 requires NUM_FRAMES=49." >&2
    exit 2
fi
if [[ "${ALLOW_CUSTOM_INFERENCE:-false}" != "true" ]]; then
    if [[ "$NUM_INFERENCE_STEPS" != "50" ]]; then
        echo "Stage 4 requires NUM_INFERENCE_STEPS=50." >&2
        exit 2
    fi
    if [[ "$SIGMA_SHIFT" != "5.0" && "$SIGMA_SHIFT" != "5" ]]; then
        echo "Stage 4 requires SIGMA_SHIFT=5.0." >&2
        exit 2
    fi
    if [[ "$CFG_SCALE" != "5.0" && "$CFG_SCALE" != "5" ]]; then
        echo "Stage 4 requires CFG_SCALE=5.0." >&2
        exit 2
    fi
fi

# Validate all variant selectors before starting any work.
python3 - "$PHASES" <<'PY'
import math, sys
for phases in sys.argv[1:]:
    seen = set()
    for phase in phases.split():
        if phase in seen:
            raise ValueError(f'Duplicate phase: {phase}')
        seen.add(phase)
        if phase in ('baseline', 'matching_only'):
            continue
        if not phase.startswith('t'):
            raise ValueError(f'Unknown phase: {phase}')
        value = float(phase[1:])
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f'Invalid threshold: {phase}')
PY

SOURCE_STEM="$(basename "$TARGET")"
SOURCE_STEM="${SOURCE_STEM%.*}"
SOURCE_STEM="${SOURCE_STEM%_splits_manifest}"
BASELINE_MANIFEST="$BASELINE_SPLITS_DIR/${SOURCE_STEM}_splits_manifest.json"
FLOWLONG_MANIFEST="$FLOWLONG_SPLITS_DIR/${SOURCE_STEM}_splits_manifest.json"
if [[ "$DRY_RUN" != "true" && ! -f "$BASELINE_MANIFEST" ]]; then
    echo "Missing baseline manifest: $BASELINE_MANIFEST" >&2
    exit 1
fi
if [[ "$DRY_RUN" != "true" && ! -f "$FLOWLONG_MANIFEST" ]]; then
    echo "Missing FlowLong manifest: $FLOWLONG_MANIFEST" >&2
    exit 1
fi

VISTA4D_CHECKPOINT=${VISTA4D_CHECKPOINT:-$VISTA4D_FOLDER}
if [[ "$DRY_RUN" != "true" ]]; then
    VISTA4D_CHECKPOINT=$(python3 -m utils.vista4d_checkpoint resolve "$VISTA4D_CHECKPOINT")
fi
PROMPT=${PROMPT:-"A realistic handheld smartphone video of people in an everyday scene, with natural body motion, realistic lighting, stable camera motion, and detailed surroundings."}

EVAL_ROOT=${EVAL_ROOT:-$EVAL_ROOT_BASE/${SOURCE_STEM}_${RESOLUTION}_seed=${SEED}}
BASELINE_OUTPUT="$EVAL_ROOT/baseline"
MATCHING_OUTPUT="$EVAL_ROOT/matching_only"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$LOG_ROOT/${SOURCE_STEM}_${RESOLUTION}_seed=${SEED}_$RUN_ID"
if [[ "$DRY_RUN" != "true" ]]; then
    mkdir -p "$EVAL_ROOT" "$LOG_DIR"
fi

phase_requested() {
    local wanted="$1"
    local phase
    for phase in $PHASES; do
        if [[ "$phase" == "$wanted" ]]; then
            return 0
        fi
    done
    return 1
}

run_logged() {
    local name="$1"
    shift
    echo
    echo "==== Stage 4: $name ===="
    if [[ "$DRY_RUN" == "true" ]]; then
        printf 'DRY_RUN:'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
}

echo "FlowLong stage-4 experiment:"
echo "    TARGET=$SOURCE_STEM"
echo "    PHASES=$PHASES"
echo "    EVAL_ROOT=$EVAL_ROOT"
echo "    BASELINE_MANIFEST=$BASELINE_MANIFEST"
echo "    FLOWLONG_MANIFEST=$FLOWLONG_MANIFEST"
echo "    RESOLUTION=$RESOLUTION ($WIDTH x $HEIGHT)"
echo "    VISTA4D_FOLDER=$VISTA4D_FOLDER"
echo "    BASELINE_EXISTING_CLIP_ROOT=${BASELINE_EXISTING_CLIP_ROOT:-<fresh generation>}"
echo "    BASELINE_EXISTING_LOG_DIR=${BASELINE_EXISTING_LOG_DIR:-<none>}"
echo "    SEED=$SEED"
echo "    STEPS=$NUM_INFERENCE_STEPS"
echo "    SIGMA_SHIFT=$SIGMA_SHIFT"
echo "    CFG_SCALE=$CFG_SCALE"
echo "    FLOWLONG_MICROBATCH_SIZE=$FLOWLONG_MICROBATCH_SIZE"
echo "    FORCE=$FORCE"
echo "    DRY_RUN=$DRY_RUN"
echo "    LOG_DIR=$LOG_DIR"
if [[ "$RESOLUTION" == "384p" ]]; then
    echo "    NOTE=GB10 measured 384p time on 310 frames: 11.90-12.66h per FlowLong variant."
else
    echo "    NOTE=GB10 measured 720p time on 310 frames: 88.45h for t*=0.6; run smoke before the rest of the matrix."
fi

if [[ "$DRY_RUN" == "true" ]]; then
    VISTA4D_CHECKPOINT_SHA256="DRY_RUN_SHA256"
else
    echo "Computing Vista4D checkpoint SHA-256 once for the complete experiment..."
    VISTA4D_CHECKPOINT_SHA256="$(python3 -m utils.vista4d_checkpoint sha256 "$VISTA4D_CHECKPOINT")"
fi

COMMON_BASELINE_ARGS=(
    --manifest "$BASELINE_MANIFEST"
    --condition_root "$BASELINE_CONDITION_ROOT"
    --render_folder "render_${RESOLUTION}_smooth"
    --existing_inference_folder "vista4d_${RESOLUTION}_smooth"
    --output_folder "$BASELINE_OUTPUT"
    --resolution "$RESOLUTION"
    --model_id_with_origin_paths "$WAN_PATHS"
    --tokenizer_id_with_origin_path "$TOKENIZER_PATHS"
    --local_model_folder "$LOCAL_WAN_FOLDER"
    --vista4d_checkpoint "$VISTA4D_CHECKPOINT"
    --vista4d_checkpoint_sha256 "$VISTA4D_CHECKPOINT_SHA256"
    --vista4d_config_path "$VISTA4D_FOLDER/config.yaml"
    --prompt "$PROMPT"
    --height "$HEIGHT"
    --width "$WIDTH"
    --num_frames "$NUM_FRAMES"
    --seed "$SEED"
    --num_inference_steps "$NUM_INFERENCE_STEPS"
    --sigma_shift "$SIGMA_SHIFT"
    --cfg_scale "$CFG_SCALE"
)
if [[ "${ALLOW_CUSTOM_INFERENCE:-false}" == "true" ]]; then
    COMMON_BASELINE_ARGS+=(--allow_custom_inference)
fi
if [[ "$TILE_VAE" == "true" ]]; then
    COMMON_BASELINE_ARGS+=(--tile_vae)
fi
if [[ "$FORCE" == "true" ]]; then
    COMMON_BASELINE_ARGS+=(--overwrite)
fi
if [[ -n "${NEGATIVE_PROMPT:-}" ]]; then
    COMMON_BASELINE_ARGS+=(--negative_prompt "$NEGATIVE_PROMPT")
fi
if [[ -n "${VRAM_LIMIT:-}" ]]; then
    COMMON_BASELINE_ARGS+=(--vram_limit "$VRAM_LIMIT")
fi
if [[ -n "$BASELINE_EXISTING_CLIP_ROOT" ]]; then
    COMMON_BASELINE_ARGS+=(--existing_clip_root "$BASELINE_EXISTING_CLIP_ROOT")
fi
if [[ -n "$BASELINE_EXISTING_LOG_DIR" ]]; then
    COMMON_BASELINE_ARGS+=(--existing_log_dir "$BASELINE_EXISTING_LOG_DIR")
fi

if phase_requested baseline; then
    run_logged baseline python3 -m scripts.inference.inference_split_baseline \
        "${COMMON_BASELINE_ARGS[@]}"
fi

run_flowlong_variant() {
    local name="$1"
    local output="$2"
    local threshold="$3"
    local disable_stochastic="$4"
    run_logged "$name" env \
        FLOWLONG_SPLITS_DIR="$FLOWLONG_SPLITS_DIR" \
        FLOWLONG_RESULT_ROOT="$FLOWLONG_RESULT_ROOT" \
        OUTPUT_FOLDER="$output" \
        RESOLUTION="$RESOLUTION" \
        HEIGHT="$HEIGHT" \
        WIDTH="$WIDTH" \
        NUM_FRAMES="$NUM_FRAMES" \
        NUM_INFERENCE_STEPS="$NUM_INFERENCE_STEPS" \
        SIGMA_SHIFT="$SIGMA_SHIFT" \
        CFG_SCALE="$CFG_SCALE" \
        FLOWLONG_STOCHASTIC_THRESHOLD="$threshold" \
        FLOWLONG_DISABLE_STOCHASTIC="$disable_stochastic" \
        FLOWLONG_MICROBATCH_SIZE="$FLOWLONG_MICROBATCH_SIZE" \
        SEEDS="$SEED" \
        PROMPT="$PROMPT" \
        TILE_VAE="$TILE_VAE" \
        FORCE="$FORCE" \
        LOCAL_WAN_FOLDER="$LOCAL_WAN_FOLDER" \
        VISTA4D_FOLDER="$VISTA4D_FOLDER" \
        VISTA4D_CHECKPOINT="$VISTA4D_CHECKPOINT" \
        VISTA4D_CHECKPOINT_SHA256="$VISTA4D_CHECKPOINT_SHA256" \
        bash scripts/test_video/run_flowlong_inference.sh "$SOURCE_STEM"
}

if phase_requested matching_only; then
    run_flowlong_variant matching_only "$MATCHING_OUTPUT" 0.6 true
fi
for phase in $PHASES; do
    if [[ "$phase" == t* ]]; then
        threshold=${phase#t}
        variant=$(python3 -c 'from utils.video_config import threshold_name; import sys; print(threshold_name(float(sys.argv[1])))' "$threshold")
        run_flowlong_variant "$variant" "$EVAL_ROOT/$variant" "$threshold" false
    fi
done

echo
echo "Stage-4 requested phases finished."
echo "Experiment root: $EVAL_ROOT"
echo "Logs: $LOG_DIR"
