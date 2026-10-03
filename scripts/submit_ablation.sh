#!/bin/bash
set -euo pipefail
export PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cluster="${1:-dgx}"
shift || true
variant="${1:?usage: submit_ablation.sh dgx|selena same_user|full_datastore|raw_distance|no_query_scaling [Hydra overrides]}"
shift
export SELECTIME_STUDY=ablation SELECTIME_VARIANT="$variant"
source "$PROJECT_ROOT/src/slurm/submit_study.sh" "$cluster" "$@" "variant=$variant"
