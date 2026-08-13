from types import SimpleNamespace

import pandas as pd
import yaml

from train import (
    main,
    write_calibration_decision,
    write_confirmatory_decision,
)


def test_validate_only_writes_map_gate_without_training(tmp_path):
    output = tmp_path / "protocol"
    main(
        [
            "--config",
            str(tmp_path / "missing.yaml"),
            "--grid_size",
            "4",
            "--max_steps",
            "4",
            "--survival_reward",
            "0.5",
            "--terminal_survival_bonus",
            "1.0",
            "--random_start",
            "--boundary_mines",
            "--num_train_maps",
            "2",
            "--num_validation_maps",
            "1",
            "--num_benchmark_maps",
            "2",
            "--trivial_policy_rollouts",
            "2",
            "--max_constant_success",
            "0.0",
            "--minimum_safe_heuristic_success",
            "0.1",
            "--results-dir",
            str(output / "results"),
            "--checkpoints-dir",
            str(output / "checkpoints"),
            "--videos-dir",
            str(output / "videos"),
            "--train-map-dir",
            str(output / "train"),
            "--validation-map-dir",
            str(output / "validation"),
            "--benchmark-map-dir",
            str(output / "test"),
            "--validate-only",
        ]
    )

    summary = pd.read_csv(output / "results" / "map_validity_summary.csv")
    constants = summary[summary["Policy"].str.startswith("constant_")]
    safe = summary[summary["Policy"] == "safe_heuristic"]
    assert (constants["Success_Mean"] == 0).all()
    assert set(safe["Split"]) == {"train", "validation", "test"}
    assert (safe["Success_Mean"] >= 0.1).all()
    assert not (output / "results" / "training_runs.csv").exists()


def test_calibration_requires_every_variant_to_escape_floor(tmp_path):
    runs = pd.DataFrame(
        [
            {"Variant": "ppo", "Test Success Mean": 0.2},
            {"Variant": "icm_planner", "Test Success Mean": 0.0},
        ]
    )
    args = SimpleNamespace(
        results_dir=str(tmp_path),
        minimum_calibration_test_success=0.05,
        variants=["ppo", "icm_planner"],
    )

    decision = write_calibration_decision(runs, args)

    assert decision["calibration_passed"] is False

    incomplete = write_calibration_decision(runs.iloc[:1], args)
    assert incomplete["calibration_passed"] is False
    assert incomplete["missing_variants"] == ["icm_planner"]


def test_confirmatory_config_locks_the_declared_hybrid_contract():
    with open("configs/confirmatory.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    assert config["variants"] == [
        "ppo",
        "icm",
        "rnd",
        "pc",
        "planner",
        "icm_planner",
    ]
    assert config["primary_variant"] == "icm_planner"
    assert config["num_seeds"] == 40
    assert (
        config["num_train_maps"],
        config["num_validation_maps"],
        config["num_benchmark_maps"],
    ) == (30, 10, 30)
    assert config["evaluation_rollouts"] == 20
    assert config["minimum_success_effect"] == 0.05
    assert config["map_seed_base"] != 50_000


def test_confirmatory_claim_requires_all_five_comparators(tmp_path):
    comparators = ["ppo", "icm", "rnd", "pc", "planner"]
    success_tests = pd.DataFrame(
        {
            "Comparator": comparators,
            "Superiority Criterion Met": [True] * 5,
        }
    )
    args = SimpleNamespace(
        results_dir=str(tmp_path),
        primary_variant="icm_planner",
        minimum_success_effect=0.05,
        minimum_reward_effect=0.5,
    )

    supported = write_confirmatory_decision(success_tests, args, comparators)
    assert supported["primary_claim_supported"] is True
    assert supported["holm_family_size"] == 5

    failed_comparison = success_tests.copy()
    failed_comparison.loc[
        failed_comparison["Comparator"] == "ppo",
        "Superiority Criterion Met",
    ] = False
    unsupported = write_confirmatory_decision(
        failed_comparison,
        args,
        comparators,
    )
    assert unsupported["primary_claim_supported"] is False

    incomplete = write_confirmatory_decision(
        success_tests.iloc[:-1],
        args,
        comparators,
    )
    assert incomplete["primary_claim_supported"] is False
    assert incomplete["missing_comparators"] == ["planner"]
