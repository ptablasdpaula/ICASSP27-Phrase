#!/usr/bin/env bash
# Publish only this study's validated reports, under the user's standing auto-push instruction.
set -euo pipefail
cd /data/home/acw794/ICASSP27-Phrase
export PYTHONHOME="$PWD/.pixi/envs/default" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
.pixi/envs/default/bin/python scripts/report_direction_recovery.py
.pixi/envs/default/bin/ruff check --no-cache --no-respect-gitignore \
  scripts/report_direction_recovery.py scripts/run_direction_recovery.py \
  scripts/_recovery_study_fit.py scripts/qualify_direction_recovery.py \
  scripts/run_clockwise_recovery.py scripts/_clockwise_study_fit.py \
  scripts/qualify_clockwise_recovery.py \
  scripts/run_fading_recovery.py scripts/qualify_fading_recovery.py
test "$(git branch --show-current)" = main
git pull --ff-only
git add -- docs/direction-recovery-150
git diff --cached --check -- docs/direction-recovery-150
if ! git diff --cached --quiet -- docs/direction-recovery-150; then
  git commit --only -m "Report full orthogonal clockwise and fixed-fading recovery study" -- docs/direction-recovery-150
fi
git push origin HEAD:main
