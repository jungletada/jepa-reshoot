#!/usr/bin/env bash
# No 'set -u': conda's activate.d hooks reference unbound vars and would abort the run.
set -o pipefail
cd /home/p50057753/workspace/code/vista4d
source "/home/p50057753/miniforge3/etc/profile.d/conda.sh"
conda activate vista4d-pgx
echo "=== started $(date -Is) on $(hostname) ==="
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
RESOLUTION=384p \
PHASES="baseline matching_only t0.5 t0.6 t0.7" \
bash scripts/test_video/run_flowlong_stage4.sh 1778135019043
status=$?
echo "=== finished $(date -Is) with exit status $status ==="
