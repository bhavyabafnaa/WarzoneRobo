import pandas as pd

from train import main


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
