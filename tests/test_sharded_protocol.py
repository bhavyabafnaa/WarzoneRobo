from pathlib import Path

import pandas as pd
import yaml

from scripts.merge_confirmatory_shards import merge_confirmatory_shards
from train import VARIANTS, main, parse_args, resolve_seeds


def _smoke_config(tmp_path: Path) -> Path:
    config = {
        "grid_size": 4,
        "num_episodes": 2,
        "max_steps": 4,
        "survival_reward": 0.5,
        "terminal_survival_bonus": 1.0,
        "dynamic_risk": True,
        "dynamic_cost": True,
        "random_start": True,
        "boundary_mines": True,
        "initial_training_horizon": 2,
        "map_seed_base": 1234,
        "num_seeds": 2,
        "variants": list(VARIANTS),
        "num_train_maps": 2,
        "num_validation_maps": 1,
        "num_benchmark_maps": 1,
        "evaluation_seed": 5000,
        "evaluation_rollouts": 1,
        "greedy_evaluation_rollouts": 1,
        "trivial_policy_rollouts": 1,
        "max_constant_success": 1.0,
        "minimum_safe_heuristic_success": 0.0,
        "minimum_reward_effect": 0.0,
        "minimum_success_effect": 0.0,
        "minimum_calibration_test_success": 0.0,
        "primary_variant": "icm_planner",
        "stat_test": "paired",
        "log_backend": "none",
    }
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _run_smoke_shard(config: Path, shards: Path, seed: int) -> None:
    output = shards / f"confirmatory-seed-{seed}"
    arguments = [
        "--config",
        str(config),
        "--seeds",
        str(seed),
        "--defer-analysis",
        "--train-map-dir",
        str(output / "maps" / "train"),
        "--validation-map-dir",
        str(output / "maps" / "validation"),
        "--benchmark-map-dir",
        str(output / "maps" / "test"),
        "--results-dir",
        str(output / "results"),
        "--checkpoints-dir",
        str(output / "checkpoints"),
        "--plot-dir",
        str(output / "plots"),
        "--videos-dir",
        str(output / "videos"),
    ]
    if seed == 0:
        arguments.append("--render-videos")
    main(arguments)
    (output / "run.log").write_text(f"completed seed {seed}\n", encoding="utf-8")


def test_explicit_seed_overrides_yaml_num_seeds(tmp_path):
    config = _smoke_config(tmp_path)
    args = parse_args(["--config", str(config), "--seeds", "7"])
    assert args.num_seeds == 2
    assert resolve_seeds(args) == [7]


def test_two_seed_shards_merge_into_complete_outputs(tmp_path):
    config = _smoke_config(tmp_path)
    shards = tmp_path / "downloaded-shards"
    _run_smoke_shard(config, shards, seed=0)
    _run_smoke_shard(config, shards, seed=1)

    output = tmp_path / "merged"
    inventory = merge_confirmatory_shards(
        config,
        shards,
        output,
        require_videos=True,
    )

    assert inventory["status"] == "complete"
    assert inventory["trained_policies"] == 12
    assert inventory["training_episode_rows"] == 24
    assert inventory["evaluation_episode_rows"] == 36
    assert inventory["checkpoints"] == 12
    assert inventory["maps"] == 4
    assert inventory["videos"] == 6
    assert inventory["logs"] == 2

    results = output / "results"
    assert len(pd.read_csv(results / "test_runs.csv")) == 12
    assert len(pd.read_csv(results / "confirmatory_tests.csv")) == 10
    assert len(pd.read_csv(results / "interaction_tests.csv")) == 2
    assert (results / "confirmatory_decision.yaml").exists()
    assert (results / "result_inventory.yaml").exists()
    assert len(list((output / "plots").glob("*.pdf"))) == 6
