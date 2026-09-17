#!/usr/bin/env bash
# sbatch --cpus-per-task=4 --mem=16G --time=04:00:00 jobs/gradient_assessment.sh
set -euo pipefail
cd /data/home/acw794/ICASSP27-Phrase
export PYTHONHOME="$PWD/.pixi/envs/default" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python_bin="$PWD/.pixi/envs/default/bin/python"
for batch in 4 8 16; do
  "$python_bin" scripts/assess_gradients.py benchmark --batch "$batch"
done
batch=$($python_bin - <<'PY'
import json
from pathlib import Path
options = []
for batch in (4, 8, 16):
    rows = json.loads(Path(f'results/gradient-assessment/benchmark-b{batch}.json').read_text())
    if max(r['peak_rss_kib'] for r in rows) < 2 * 1024**2:
        # Six multi-event columns dominate; weight 2/4 events equally.
        speed = sum(r['seconds_per_candidate'] for r in rows if r['events'] > 1)
        options.append((speed, batch))
if not options:
    raise SystemExit('No qualified batch fits the 2 GiB worker budget')
print(min(options)[1])
PY
)
echo "Selected batch $batch"
"$python_bin" scripts/assess_gradients.py campaign --batch "$batch" \
  --workers "${SLURM_CPUS_PER_TASK:-4}"
"$python_bin" scripts/assess_gradients.py report
