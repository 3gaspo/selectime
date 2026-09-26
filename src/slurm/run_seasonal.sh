#!/bin/bash
set -euo pipefail
source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export SELECTIME_DEFER_COMPLETION=1
trap 'uv run --no-sync python -m timebench.scripts.finalize_stage "$TIME_SEASONAL_TASKS_ROOT" "$TIME_LAUNCH_ID" --interrupt' EXIT
srun --ntasks=1 uv run --no-sync python src/scripts/run_seasonal.py "$@" "experiment_mode=${EXPERIMENT_MODE:-full}"
uv run --no-sync python -m timebench.scripts.finalize_stage "$TIME_SEASONAL_TASKS_ROOT" "$TIME_LAUNCH_ID"
trap - EXIT
