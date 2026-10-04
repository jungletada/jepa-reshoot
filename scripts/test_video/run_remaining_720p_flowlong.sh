#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISTA4D_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$VISTA4D_ROOT"

TARGET=${1:-${TARGET:-}}
RESOLUTION=${RESOLUTION:-720p}
SEED=${SEED:-10027}
FLOWLONG_STOCHASTIC_THRESHOLD=${FLOWLONG_STOCHASTIC_THRESHOLD:-0.6}
STATUS_FILE=${STATUS_FILE:-./logs/pipeline/${TARGET}_${RESOLUTION}_remaining.status}

if [[ -z "$TARGET" ]]; then
    echo "Usage: bash scripts/test_video/run_remaining_720p_flowlong.sh <source_stem>" >&2
    exit 2
fi
if [[ "$RESOLUTION" != "720p" ]]; then
    echo "This runner is intentionally restricted to RESOLUTION=720p." >&2
    exit 2
fi

BASELINE_SPLITS_DIR=${BASELINE_SPLITS_DIR:-./media/splits/720p}
FLOWLONG_SPLITS_DIR=${FLOWLONG_SPLITS_DIR:-./media/flowlong_splits/720p}
BASELINE_MANIFEST="$BASELINE_SPLITS_DIR/${TARGET}_splits_manifest.json"
FLOWLONG_MANIFEST="$FLOWLONG_SPLITS_DIR/${TARGET}_splits_manifest.json"
FULL_SEQUENCE_ROOT=${FULL_SEQUENCE_ROOT:-./results/full/${TARGET}_stitched_${RESOLUTION}}
BASELINE_CONDITION_ROOT=${BASELINE_CONDITION_ROOT:-./results/shared_static_single}
FLOWLONG_RESULT_ROOT=${FLOWLONG_RESULT_ROOT:-./results/flowlong_single}
SMOKE_OUTPUT=${SMOKE_OUTPUT:-./results/flowlong_smoke/${TARGET}_${RESOLUTION}_1step}
FORMAL_OUTPUT=${FORMAL_OUTPUT:-./results/flowlong_eval/${TARGET}_${RESOLUTION}_seed=${SEED}/flowlong_t0p6}

mkdir -p "$(dirname "$STATUS_FILE")"
: > "$STATUS_FILE"

mark() {
    local message="$1"
    printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$message" | tee -a "$STATUS_FILE"
}

validate_inference_output() {
    local output_folder="$1"
    local expected_steps="$2"
    python3 - "$FLOWLONG_MANIFEST" "$output_folder" "$SEED" "$expected_steps" <<'PY'
import json
from pathlib import Path
import sys

import cv2

manifest_path = Path(sys.argv[1])
output_folder = Path(sys.argv[2])
seed = int(sys.argv[3])
expected_steps = int(sys.argv[4])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
report_path = output_folder / f"flowlong_report_seed={seed}.json"
video_path = output_folder / f"video_seed={seed}.mp4"
report = json.loads(report_path.read_text(encoding="utf-8"))

expected_frames = int(manifest["end_exclusive"]) - int(manifest["start_frame"])
expected_windows = len(manifest["clips"])
assert report["output_frames"] == expected_frames, report["output_frames"]
assert report["pipeline"]["num_windows"] == expected_windows
assert report["pipeline"]["num_inference_steps"] == expected_steps
assert report["pipeline"]["valid_output_frames"] == expected_frames
assert report["pipeline"]["trimmed_frames"] == report["geometry"]["trim_right"]
assert all(step["overlap_after_max_abs"] == 0.0 for step in report["pipeline"]["steps"])

cap = cv2.VideoCapture(str(video_path))
assert cap.isOpened(), video_path
observed = (
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
)
cap.release()
assert observed == (1280, 720, expected_frames), observed
print(
    f"Inference validation: PASS output={output_folder}, frames={expected_frames}, "
    f"windows={expected_windows}, steps={expected_steps}, video={observed}"
)
PY
}

mark "PREFLIGHT_START"
export VISTA4D_FOLDER=${VISTA4D_FOLDER:-./checkpoints/vista4d/720p49_step=3000}
export VISTA4D_CHECKPOINT
VISTA4D_CHECKPOINT=$(python3 -m utils.vista4d_checkpoint resolve "${VISTA4D_CHECKPOINT:-$VISTA4D_FOLDER}")
for required in \
    "$BASELINE_MANIFEST" \
    "$FLOWLONG_MANIFEST" \
    "$VISTA4D_CHECKPOINT" \
    "$VISTA4D_FOLDER/config.yaml" \
    ./checkpoints/wan/Wan2.1-T2V-14B; do
    if [[ ! -e "$required" ]]; then
        echo "Missing required input: $required" >&2
        exit 1
    fi
done

python3 - "$BASELINE_MANIFEST" "$FLOWLONG_MANIFEST" <<'PY'
import json
from pathlib import Path
import sys

from utils.split_manifest import validate_flowlong_window_manifest

baseline = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
flowlong = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
validate_flowlong_window_manifest(baseline, clip_frames=49, overlap=5, temporal_alignment=4)
validate_flowlong_window_manifest(flowlong, clip_frames=49, overlap=25, temporal_alignment=4)
for key in ("input_path", "start_frame", "end_exclusive", "fps", "resolution"):
    assert baseline[key] == flowlong[key], (key, baseline[key], flowlong[key])
print(
    "Preflight manifests: PASS "
    f"frames={baseline['end_exclusive'] - baseline['start_frame']}, "
    f"baseline_windows={len(baseline['clips'])}, "
    f"flowlong_windows={len(flowlong['clips'])}"
)
PY
mark "PREFLIGHT_DONE"

mark "STITCH_SMOOTH_SLICE_START"
SPLITS_DIR="$BASELINE_SPLITS_DIR" \
RESOLUTION="$RESOLUTION" \
INPUT_RESULT_ROOT=./results/single \
OUTPUT_RESULT_ROOT=./results/stitched_single \
FULL_RESULT_BASE=./results/full \
TRANSLATION_SIGMA=8 \
ROTATION_SIGMA=10 \
FORCE_STITCH=true \
OVERWRITE_SPLITS=true \
bash scripts/test_video/stitch_splits_smooth_and_slice.sh "$TARGET"

python3 - "$BASELINE_MANIFEST" "$FULL_SEQUENCE_ROOT/recon_and_seg" <<'PY'
import json
from pathlib import Path
import sys

import cv2
import numpy as np

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
folder = Path(sys.argv[2])
expected = int(manifest["end_exclusive"]) - int(manifest["start_frame"])
raw = np.load(folder / "cameras.npz")
smooth = np.load(folder / "cameras_gaussian_smooth.npz")
assert raw["cam_c2w"].shape == smooth["cam_c2w"].shape == (expected, 4, 4)
assert raw["intrinsics"].shape[0] == smooth["intrinsics"].shape[0] == expected
assert len(list((folder / "depths").glob("*"))) == expected
assert len(list((folder / "dynamic_mask").glob("*"))) == expected
assert len(list((folder / "sky_mask").glob("*"))) == expected
cap = cv2.VideoCapture(str(folder / "video.mp4"))
assert cap.isOpened()
observed = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
assert observed == expected, observed
report = json.loads((folder / "stitch_report.json").read_text(encoding="utf-8"))
print(f"Stitched reconstruction validation: PASS frames={expected}, report={report.keys()}")
PY
mark "STITCH_SMOOTH_SLICE_DONE"

mark "SHARED_RENDER_AND_CONDITIONS_START"
BASELINE_SPLITS_DIR="$BASELINE_SPLITS_DIR" \
FLOWLONG_SPLITS_DIR="$FLOWLONG_SPLITS_DIR" \
BASELINE_CONDITION_ROOT="$BASELINE_CONDITION_ROOT" \
FLOWLONG_RESULT_ROOT="$FLOWLONG_RESULT_ROOT" \
FULL_SEQUENCE_ROOT="$FULL_SEQUENCE_ROOT" \
RESOLUTION="$RESOLUTION" \
STATIC_FRAME_STRIDE=4 \
RENDER_CHUNK_SIZE=4 \
OVERWRITE_FULL_RENDER=true \
OVERWRITE_BASELINE_SPLITS=true \
OVERWRITE_FLOWLONG_SPLITS=true \
bash scripts/test_video/prepare_flowlong_ab_conditions.sh "$TARGET"

mark "SHARED_RENDER_AND_CONDITIONS_DONE"

mark "FLOWLONG_SMOKE_START"
RESOLUTION="$RESOLUTION" \
FLOWLONG_SPLITS_DIR="$FLOWLONG_SPLITS_DIR" \
FLOWLONG_RESULT_ROOT="$FLOWLONG_RESULT_ROOT" \
OUTPUT_FOLDER="$SMOKE_OUTPUT" \
NUM_INFERENCE_STEPS=1 \
CFG_SCALE=1.0 \
SIGMA_SHIFT=5.0 \
FLOWLONG_STOCHASTIC_THRESHOLD="$FLOWLONG_STOCHASTIC_THRESHOLD" \
FLOWLONG_MICROBATCH_SIZE=1 \
FLOWLONG_DISABLE_STOCHASTIC=false \
SEEDS="$SEED" \
TILE_VAE=true \
USE_USP=false \
CFG_MERGE=false \
FORCE=true \
bash scripts/test_video/run_flowlong_inference.sh "$TARGET"
validate_inference_output "$SMOKE_OUTPUT" 1
mark "FLOWLONG_SMOKE_DONE"

mark "FLOWLONG_50STEP_T0P6_START"
RESOLUTION="$RESOLUTION" \
FLOWLONG_SPLITS_DIR="$FLOWLONG_SPLITS_DIR" \
FLOWLONG_RESULT_ROOT="$FLOWLONG_RESULT_ROOT" \
OUTPUT_FOLDER="$FORMAL_OUTPUT" \
NUM_INFERENCE_STEPS=50 \
CFG_SCALE=5.0 \
SIGMA_SHIFT=5.0 \
FLOWLONG_STOCHASTIC_THRESHOLD="$FLOWLONG_STOCHASTIC_THRESHOLD" \
FLOWLONG_MICROBATCH_SIZE=1 \
FLOWLONG_DISABLE_STOCHASTIC=false \
SEEDS="$SEED" \
TILE_VAE=true \
USE_USP=false \
CFG_MERGE=false \
FORCE=true \
bash scripts/test_video/run_flowlong_inference.sh "$TARGET"
validate_inference_output "$FORMAL_OUTPUT" 50
mark "FLOWLONG_50STEP_T0P6_DONE"
mark "PIPELINE_COMPLETE"
