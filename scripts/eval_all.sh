#!/usr/bin/env bash
set -euo pipefail

# Evaluate all checkpoints in a deterministic order.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="$SCRIPT_DIR/.."
cd "$REPO_ROOT"

export PYTHONHASHSEED=0
CHECKPOINT_DIR="checkpoints"

# Sort checkpoints to ensure deterministic evaluation order
readarray -t CHECKPOINTS < <(find "$CHECKPOINT_DIR" -maxdepth 1 -name '*.pt' | sort)

for ckpt in "${CHECKPOINTS[@]}"; do
  echo "=== Evaluating $(basename "$ckpt") ==="
  python - <<'PY'
import sys, yaml
from pathlib import Path
from src.env import GridWorldICM, evaluate_on_benchmarks
from src.ppo import PPOPolicy
from src.icm import ICMModule
from src.rnd import RNDModule
from src.utils import load_model

ckpt_path = Path(sys.argv[1])
# Load configuration to construct the environment
cfg = yaml.safe_load(Path('configs/default.yaml').read_text())
env = GridWorldICM(
    grid_size=cfg.get('grid_size', 8),
    dynamic_risk=cfg.get('dynamic_risk', True),
    dynamic_cost=cfg.get('dynamic_cost', True),
    survival_reward=cfg.get('survival_reward', 0.05),
    mine_density_range=cfg.get('mine_density_range'),
    hazard_density_range=cfg.get('hazard_density_range'),
    enemy_speed_range=cfg.get('enemy_speed_range'),
    enemy_policy_options=cfg.get('enemy_policies')
)
obs, _ = env.reset()
input_dim = obs.size + 2
action_dim = 4
policy, _, _ = load_model(PPOPolicy, input_dim, action_dim, str(ckpt_path), ICMModule, RNDModule)
mean, ci = evaluate_on_benchmarks(env, policy, 'test_maps', 5, H=cfg.get('H', 8))
print(f"{ckpt_path.name}: reward={mean:.2f} +/- {ci:.2f}")
PY

done
