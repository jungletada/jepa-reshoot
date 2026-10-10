#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VISTA4D_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$VISTA4D_ROOT"

TARGET=${1:-${TARGET:-}}
RESOLUTION=384p
WIDTH=672
HEIGHT=384
BASELINE_SPLITS_DIR=${BASELINE_SPLITS_DIR:-./media/splits}
FLOWLONG_SPLITS_DIR=${FLOWLONG_SPLITS_DIR:-./media/flowlong_splits}
FULL_SEQUENCE_ROOT=${FULL_SEQUENCE_ROOT:-./results/full/${TARGET}_stitched_${RESOLUTION}}
BASELINE_CONDITION_ROOT=${BASELINE_CONDITION_ROOT:-./results/shared_static_single}
FLOWLONG_RESULT_ROOT=${FLOWLONG_RESULT_ROOT:-./results/flowlong_single}
STATUS_FILE=${STATUS_FILE:-./logs/pipeline/${TARGET}_${RESOLUTION}_prepare.status}

if [[ -z "$TARGET" ]]; then
    echo "Usage: bash scripts/test_video/run_prepare_384p_flowlong.sh <source_stem>" >&2
    exit 2
fi

VIDEO=./data/${TARGET}.mp4
BASELINE_MANIFEST="$BASELINE_SPLITS_DIR/${TARGET}_splits_manifest.json"
FLOWLONG_MANIFEST="$FLOWLONG_SPLITS_DIR/${TARGET}_splits_manifest.json"

mkdir -p "$(dirname "$STATUS_FILE")"
: > "$STATUS_FILE"

mark() {
    local message="$1"
    printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$message" | tee -a "$STATUS_FILE"
}

mark "PREFLIGHT_START"
for required in \
    "$VIDEO" \
    ./checkpoints/DA3NESTED-GIANT-LARGE-1.1 \
    ./checkpoints/sam3/sam3.pt; do
    if [[ ! -e "$required" ]]; then
        echo "Missing required input: $required" >&2
        exit 1
    fi
done
mark "PREFLIGHT_DONE"

mark "SPLIT_384P_START"
OUTPUT_DIR="$BASELINE_SPLITS_DIR" \
RESOLUTION="$RESOLUTION" \
CLIP_FRAMES=49 \
OVERLAP=5 \
TEMPORAL_ALIGNMENT=4 \
INCLUDE_TAIL=true \
bash scripts/test_video/split_video.sh "$VIDEO"

OUTPUT_DIR="$FLOWLONG_SPLITS_DIR" \
RESOLUTION="$RESOLUTION" \
CLIP_FRAMES=49 \
OVERLAP=25 \
TEMPORAL_ALIGNMENT=4 \
INCLUDE_TAIL=true \
bash scripts/test_video/split_video.sh "$VIDEO"

python3 - "$BASELINE_MANIFEST" "$FLOWLONG_MANIFEST" <<'PY'
import json
from pathlib import Path
import sys

from utils.resolution import validate_manifest_resolution
from utils.split_manifest import validate_flowlong_window_manifest

baseline = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
flowlong = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
for manifest in (baseline, flowlong):
    validate_manifest_resolution(
        manifest, resolution="384p", height=384, width=672
    )
validate_flowlong_window_manifest(
    baseline, clip_frames=49, overlap=5, temporal_alignment=4
)
validate_flowlong_window_manifest(
    flowlong, clip_frames=49, overlap=25, temporal_alignment=4
)
for key in ("input_path", "start_frame", "end_exclusive", "fps"):
    assert baseline[key] == flowlong[key], (key, baseline[key], flowlong[key])
frame_count = int(baseline["end_exclusive"]) - int(baseline["start_frame"])
assert frame_count > 0, frame_count
assert baseline["clips"], "baseline manifest has no clips"
assert flowlong["clips"], "FlowLong manifest has no clips"
print(
    "384p manifests: PASS "
    f"frames={frame_count}, "
    f"baseline_clips={len(baseline['clips'])}, "
    f"flowlong_clips={len(flowlong['clips'])}, "
    f"baseline_starts={[c['start_frame'] for c in baseline['clips']]}, "
    f"flowlong_starts={[c['start_frame'] for c in flowlong['clips']]}"
)
PY
mark "SPLIT_384P_DONE"

mark "DA3_SAM3_384P_START"
export SEG_KEYWORDS="person man woman hand phone bag backpack car stroller"
while IFS=, read -r _ _ _ _ clip _; do
    if [[ -z "$clip" ]]; then
        continue
    fi
    echo "==== DA3_SAM3_START $clip ===="
    SOURCE_VIDEO="$clip" \
    RECON_METHOD=da3 \
    RESOLUTION="$RESOLUTION" \
    DA3_PROCESS_RES="$WIDTH" \
    SAVE_VIS=false \
    bash scripts/test_video/recon_and_seg.sh
    echo "==== DA3_SAM3_DONE $clip ===="
done < <(tail -n +2 "$BASELINE_SPLITS_DIR/${TARGET}_splits_manifest.csv")

python3 - "$BASELINE_MANIFEST" <<'PY'
import json
from pathlib import Path
import sys

import cv2
import numpy as np

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for clip in manifest["clips"]:
    example = Path(clip["output_path"]).stem
    folder = Path("results/single") / example / "recon_and_seg"
    required = (
        folder / "video.mp4",
        folder / "cameras.npz",
        folder / "depths",
        folder / "dynamic_mask",
        folder / "sky_mask",
    )
    assert all(path.exists() for path in required), (example, required)
    cap = cv2.VideoCapture(str(folder / "video.mp4"))
    assert cap.isOpened(), folder / "video.mp4"
    observed = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    assert observed == (672, 384, 49), (example, observed)
    assert len(list((folder / "depths").glob("*"))) == 49
    assert len(list((folder / "dynamic_mask").glob("*"))) == 49
    assert len(list((folder / "sky_mask").glob("*"))) == 49
    cameras = np.load(folder / "cameras.npz")
    assert cameras["cam_c2w"].shape == (49, 4, 4)
    assert cameras["intrinsics"].shape[0] == 49
    print(f"DA3/SAM3 validation: PASS {example}")
PY
mark "DA3_SAM3_384P_DONE"

mark "STITCH_SMOOTH_SLICE_384P_START"
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
cap = cv2.VideoCapture(str(folder / "video.mp4"))
assert cap.isOpened()
observed = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
assert observed == expected, observed
assert (folder / "stitch_report.json").is_file()
print(f"384p stitched reconstruction: PASS frames={expected}")
PY
mark "STITCH_SMOOTH_SLICE_384P_DONE"

mark "SHARED_RENDER_CONDITIONS_384P_START"
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

mark "SHARED_RENDER_CONDITIONS_384P_DONE"
mark "PIPELINE_COMPLETE"
