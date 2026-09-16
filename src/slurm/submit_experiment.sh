#!/bin/bash
set -euo pipefail
cd "$PROJECT_ROOT"
cluster="${1:-dgx}"
export OUTPUTS_ROOT="$PROJECT_ROOT/outputs" LOGS_ROOT="$PROJECT_ROOT/logs"
export TIME_OUTPUTS="$OUTPUTS_ROOT" TIME_LOGS="$LOGS_ROOT"
case "$cluster" in
    dgx)
        export TIME_STORAGE_ROOT="${TIME_STORAGE_ROOT:-$HOME}"
        source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
        front=selectime.slurm ;;
    selena)
        source "$PROJECT_ROOT/src/slurm/selena_runtime.sh"
        front=selectime_selena.slurm ;;
    *) echo 'usage: bash scripts/submit_experiment.sh dgx|selena [Hydra overrides...]' >&2; exit 2 ;;
esac
shift || true
dependency=()
[ -z "${SBATCH_DEPENDENCY:-}" ] || dependency=(--dependency="$SBATCH_DEPENDENCY")
mkdir -p "$TIME_LOGS"
sbatch "${dependency[@]}" "$front" "$@"
