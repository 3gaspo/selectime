#!/bin/bash
set -euo pipefail
export OUTPUTS_ROOT="$PROJECT_ROOT/outputs" LOGS_ROOT="$PROJECT_ROOT/logs"
export TIME_OUTPUTS="$OUTPUTS_ROOT" TIME_LOGS="$LOGS_ROOT"
source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SELECTIME_DEFER_COMPLETION=1
stages="${STAGES:-prepare,extract_validation,predict_validation,select_task,select_per_variate,extract_test,predict_test,assemble,evaluate,report}"
mode="${EXPERIMENT_MODE:-full}"
trap 'uv run --no-sync python -m timebench.scripts.finalize_stage "$TIME_OUTPUTS/selectime" "$TIME_LAUNCH_ID" --interrupt' EXIT
IFS=',' read -r -a selected_stages <<< "$stages"
for stage in "${selected_stages[@]}"; do
    case "$stage" in
        select_task) entry=src/scripts/select_per_task.py ;;
        select_per_variate) entry=src/scripts/select_per_variate.py ;;
        prepare|extract_validation|predict_validation|extract_test|predict_test|assemble|evaluate|report) entry=src/scripts/run_selectime.py ;;
        *) echo "unknown stage: $stage" >&2; exit 2 ;;
    esac
    groups=(all)
    case "$stage" in predict_validation|predict_test) groups=(vanilla remaining) ;; esac
    for group in "${groups[@]}"; do
        echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Slurm=$SLURM_JOB_ID launch=$TIME_LAUNCH_ID stage=$stage prediction_group=$group"
        srun --ntasks=1 uv run --no-sync python "$entry" "$@" "experiment_mode=$mode" "stage=$stage" "prediction_group=$group"
        uv run --no-sync python -m timebench.scripts.finalize_stage "$TIME_OUTPUTS/selectime" "$TIME_LAUNCH_ID"
    done
done
trap - EXIT
