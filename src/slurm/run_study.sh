#!/bin/bash
set -euo pipefail
case "$SELECTIME_STUDY" in
    ablation)
        export TIME_EXPERIMENT="scope_ablation_$SELECTIME_VARIANT"
        entry=src/scripts/run_selectime_ablation.py
        default_stages=prepare,extract_validation,predict_validation,calibrate,extract_test,predict_test,assemble,evaluate,report ;;
    oracles)
        export TIME_EXPERIMENT=scope_oracles
        entry=src/scripts/run_selectime_oracles.py
        default_stages=calibrate,assemble,evaluate,report ;;
    timing)
        export TIME_EXPERIMENT=scope_timing
        entry=src/scripts/run_selectime_timing.py
        default_stages=measure,report ;;
    *) echo "unknown study: $SELECTIME_STUDY" >&2; exit 2 ;;
esac
source "$PROJECT_ROOT/src/slurm/runtime_paths.sh"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SELECTIME_DEFER_COMPLETION=1
trap 'uv run --no-sync python -m timebench.scripts.finalize_stage "$TIME_OUTPUTS/$TIME_EXPERIMENT" "$TIME_LAUNCH_ID" --interrupt' EXIT
vanilla_overrides=()
if [ -n "${TIME_VANILLA_PREDICTIONS_PATH:-}" ]; then
    vanilla_overrides+=("vanilla_predictions_path=$TIME_VANILLA_PREDICTIONS_PATH")
fi
IFS=',' read -r -a selected_stages <<< "${STAGES:-$default_stages}"
for stage in "${selected_stages[@]}"; do
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Slurm=$SLURM_JOB_ID launch=$TIME_LAUNCH_ID study=$SELECTIME_STUDY experiment=$TIME_EXPERIMENT stage=$stage"
    srun --ntasks=1 uv run --no-sync python "$entry" "${vanilla_overrides[@]}" "$@" \
        "experiment_mode=${EXPERIMENT_MODE:-full}" "stage=$stage" "prediction_group=all"
    uv run --no-sync python -m timebench.scripts.finalize_stage "$TIME_OUTPUTS/$TIME_EXPERIMENT" "$TIME_LAUNCH_ID"
done
trap - EXIT
