#!/bin/bash
set -euo pipefail
export PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cluster="${1:-dgx}"
shift || true
source "$PROJECT_ROOT/src/slurm/submit_experiment.sh" "$cluster" model=ts_icl "$@"
