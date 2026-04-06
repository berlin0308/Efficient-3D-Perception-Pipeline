#!/usr/bin/env bash
# Full fp32_amp matrix -> runs.csv, then research_summary_0405.csv
set -euo pipefail
TOOLS="/home/nas/polin/cmu-berlin/MLS/OpenPCDet/tools"
DATA="/home/nas/polin/cmu-berlin/MLS/profile_outputs/research_matrix"
PY="${PY:-/home/ubuntu/miniconda3/envs/mls/bin/python}"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/home/ubuntu/miniconda3/envs/mls}/lib:${LD_LIBRARY_PATH:-}"

cd "$TOOLS"
"$PY" collect_research_metrics.py run \
  --matrix fp32_amp \
  --fresh_runs \
  --cuda_id 0 \
  --warmup 300 \
  --steps 50 \
  --batch_size 1 \
  --workers 2 \
  --measurement-burnin-steps 20 \
  --output_root "$DATA"

"$PY" "$DATA/generate_research_summary.py" \
  --data_dir "$DATA" \
  --output_csv "$DATA/research_summary_0405.csv"

echo "[run_full_collect_then_summary_0405] OK"
