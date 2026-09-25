#!/bin/bash
set -euo pipefail
if [ -n "${SELENA_NNI:-}" ]; then
    export OUTPUTS_ROOT="$TIME_SCRATCH_ROOT/outputs" LOGS_ROOT="$TIME_SCRATCH_ROOT/logs"
else
    export OUTPUTS_ROOT="$PROJECT_ROOT/outputs" LOGS_ROOT="$PROJECT_ROOT/logs"
fi
export TIME_OUTPUTS="$OUTPUTS_ROOT" TIME_LOGS="$LOGS_ROOT"
source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SELECTIME_DEFER_COMPLETION=1
stages="${STAGES:-prepare,extract_validation,predict_validation,calibrate,extract_test,predict_test,assemble,evaluate,report}"
mode="${EXPERIMENT_MODE:-full}"
vanilla_overrides=()
if [ -n "${TIME_VANILLA_PREDICTIONS_PATH:-}" ]; then
    vanilla_overrides+=("vanilla_predictions_path=$TIME_VANILLA_PREDICTIONS_PATH")
fi
trap 'uv run --no-sync python -m timebench.scripts.finalize_stage "$TIME_OUTPUTS/selectime" "$TIME_LAUNCH_ID" --interrupt' EXIT
IFS=',' read -r -a selected_stages <<< "$stages"
for stage in "${selected_stages[@]}"; do
    case "$stage" in
        prepare|extract_validation|predict_validation|calibrate|extract_test|predict_test|assemble|evaluate|report) entry=src/scripts/run_selectime.py ;;
        *) echo "unknown stage: $stage" >&2; exit 2 ;;
    esac
    groups=(all)
    case "$stage" in predict_validation|predict_test) groups=(vanilla remaining) ;; esac
    for group in "${groups[@]}"; do
        echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Slurm=$SLURM_JOB_ID launch=$TIME_LAUNCH_ID stage=$stage prediction_group=$group"
        srun --ntasks=1 uv run --no-sync python "$entry" "${vanilla_overrides[@]}" "$@" "experiment_mode=$mode" "stage=$stage" "prediction_group=$group"
        uv run --no-sync python -m timebench.scripts.finalize_stage "$TIME_OUTPUTS/selectime" "$TIME_LAUNCH_ID"
    done
done
trap - EXIT
