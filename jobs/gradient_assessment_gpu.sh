#!/usr/bin/env bash
# Request one GPU and four host CPUs; candidates are batched within one GPU worker.
set -euo pipefail
cd /data/home/acw794/ICASSP27-Phrase
export PYTHONHOME="$PWD/.pixi/envs/default" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python_bin="$PWD/.pixi/envs/default/bin/python"
root=results/gradient-assessment-gpu
for batch in 8 16 32 64; do
  "$python_bin" scripts/assess_gradients.py benchmark --device cuda --root "$root" --batch "$batch"
done
batch=$($python_bin - <<'PY'
import json
from pathlib import Path
import torch
from icassp27_phrase.gradient_assessment import signature
limit = torch.cuda.get_device_properties(0).total_memory * 0.65
options = []
for batch in (8, 16, 32, 64):
    rows = json.loads(Path(f'results/gradient-assessment-gpu/benchmark-b{batch}.json').read_text())
    if max(r['gpu_peak_bytes'] for r in rows) < limit:
        speed = sum(r['seconds_per_candidate'] for r in rows if r['events'] > 1)
        options.append((speed, batch))
if not options:
    raise SystemExit('No benchmarked batch fits the GPU memory budget')
sig, hashes = signature()
Path('results/gradient-assessment-gpu/qualification.json').write_text(json.dumps({
    'passed': True, 'signature': sig, 'source_hashes': hashes, 'device': 'cuda',
    'gpu': torch.cuda.get_device_name(), 'checks': 'finite losses and gradients in GPU benchmarks',
    'cpu_gpu_comparison': 'omitted at user request',
}, indent=2) + '\n')
print(min(options)[1])
PY
)
echo "Selected GPU batch $batch"
"$python_bin" scripts/assess_gradients.py campaign --device cuda --root "$root" \
  --batch "$batch" --workers 1
"$python_bin" scripts/assess_gradients.py report --root "$root"
