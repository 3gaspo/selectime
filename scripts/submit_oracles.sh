#!/bin/bash
set -euo pipefail
export PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export SELECTIME_STUDY=oracles
source "$PROJECT_ROOT/src/slurm/submit_study.sh" "$@"
