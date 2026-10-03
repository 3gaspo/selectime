#!/bin/bash
set -euo pipefail
cd "$PROJECT_ROOT"
cluster="${1:-dgx}"
shift || true
case "$SELECTIME_STUDY" in
    ablation)
        case "$SELECTIME_VARIANT" in
            same_user|full_datastore|raw_distance|no_query_scaling) ;;
            *) echo "unknown ablation: $SELECTIME_VARIANT" >&2; exit 2 ;;
        esac
        export TIME_EXPERIMENT="scope_ablation_$SELECTIME_VARIANT"
        job="selectime_$SELECTIME_VARIANT" ;;
    oracles) export TIME_EXPERIMENT=scope_oracles; job=selectime_oracles ;;
    timing) export TIME_EXPERIMENT=scope_timing; job=selectime_timing ;;
    *) echo "unknown study: $SELECTIME_STUDY" >&2; exit 2 ;;
esac
case "$cluster" in
    dgx)
        export TIME_STORAGE_ROOT="${TIME_STORAGE_ROOT:-$HOME}"
        source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
        front=selectime_study.slurm ;;
    selena)
        source "$PROJECT_ROOT/src/slurm/selena_runtime.sh"
        front=selectime_study_selena.slurm
        job="s$job" ;;
    *) echo 'cluster must be dgx or selena' >&2; exit 2 ;;
esac
dependency=()
[ -z "${SBATCH_DEPENDENCY:-}" ] || dependency=(--dependency="$SBATCH_DEPENDENCY")
mkdir -p "$TIME_LOGS/$TIME_EXPERIMENT/slurm"
sbatch "${dependency[@]}" --job-name="$job" \
    --output="$TIME_LOGS/$TIME_EXPERIMENT/slurm/%x_%j.out" \
    --error="$TIME_LOGS/$TIME_EXPERIMENT/slurm/%x_%j.err" "$front" "$@"
