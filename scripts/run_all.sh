#!/usr/bin/env bash
set -euo pipefail

# Run all experiment configurations in a deterministic order.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="$SCRIPT_DIR/.."
cd "$REPO_ROOT"

export PYTHONHASHSEED=0
ENV_CONFIG="configs/env_8x8.yaml"
ALGO_DIR="configs/algo"

# Explicitly list algorithms to ensure deterministic ordering
ALGOS=(dyna lppo planner_subgoal shielded)

for algo in "${ALGOS[@]}"; do
  echo "=== Running ${algo} ==="
  python train.py --env-config "$ENV_CONFIG" --algo-config "${ALGO_DIR}/${algo}.yaml" --seed 0
  echo
done
