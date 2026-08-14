"""Validate and merge independently executed confirmatory seed shards."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.analysis import confirmatory_comparisons, factorial_interaction
from src.visualization import generate_results_table, plot_training_curves
from train import (
    VARIANTS,
    resolve_seeds,
    summarize_runs,
    write_calibration_decision,
    write_confirmatory_decision,
    write_protocol_manifest,
    write_run_view,
)
from train import (
    parse_args as parse_training_args,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge and validate confirmatory seed artifacts"
    )
    parser.add_argument("--config", default="configs/confirmatory.yaml")
    parser.add_argument("--shards-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--require-videos",
        action="store_true",
        help="Require one seed-0 GIF for every variant",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files_identical(paths: list[Path]) -> bool:
    if len(paths) < 2:
        return True
    if all(path.suffix == ".npz" for path in paths):
        with np.load(paths[0]) as reference:
            reference_values = {
                key: np.array(reference[key], copy=True) for key in reference.files
            }
        for path in paths[1:]:
            with np.load(path) as candidate:
                if set(candidate.files) != set(reference_values):
                    return False
                if any(
                    not np.array_equal(candidate[key], value)
                    for key, value in reference_values.items()
                ):
                    return False
        return True
    return len({_sha256(path) for path in paths}) == 1


def _require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {', '.join(missing)}")


def _load_shards(
    shards_dir: Path,
    expected_seeds: list[int],
    expected_variants: list[str],
    episodes_per_run: int,
    validation_episodes: int,
    stochastic_test_episodes: int,
    greedy_test_episodes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, Path]]:
    run_files = sorted(shards_dir.rglob("results/training_runs.csv"))
    if not run_files:
        raise FileNotFoundError(
            f"No results/training_runs.csv files found below {shards_dir}"
        )

    run_frames: list[pd.DataFrame] = []
    training_frames: list[pd.DataFrame] = []
    evaluation_frames: list[pd.DataFrame] = []
    shard_roots: dict[int, Path] = {}

    for run_file in run_files:
        shard_root = run_file.parent.parent
        runs = pd.read_csv(run_file)
        _require_columns(runs, {"Seed", "Variant"}, run_file)
        seeds = sorted(int(seed) for seed in runs["Seed"].unique())
        if len(seeds) != 1:
            raise ValueError(f"{run_file} must contain exactly one seed; got {seeds}")
        seed = seeds[0]
        if seed in shard_roots:
            raise ValueError(f"Duplicate shard found for seed {seed}")
        variants = list(runs["Variant"])
        if sorted(variants) != sorted(expected_variants):
            raise ValueError(
                f"Seed {seed} variants are {sorted(variants)}, expected "
                f"{sorted(expected_variants)}"
            )

        training_file = shard_root / "results" / "training_episodes.csv"
        evaluation_file = shard_root / "results" / "evaluation_episodes.csv"
        if not training_file.exists() or not evaluation_file.exists():
            raise FileNotFoundError(f"Seed {seed} is missing episode-level CSV files")
        training = pd.read_csv(training_file)
        evaluation = pd.read_csv(evaluation_file)
        _require_columns(training, {"Seed", "Variant", "Episode"}, training_file)
        _require_columns(
            evaluation,
            {"Seed", "Variant", "Split", "Evaluation Mode"},
            evaluation_file,
        )

        training_counts = training.groupby("Variant").size()
        for variant in expected_variants:
            count = int(training_counts.get(variant, 0))
            if count != episodes_per_run:
                raise ValueError(
                    f"Seed {seed}/{variant} has {count} training episodes; "
                    f"expected {episodes_per_run}"
                )

        expected_evaluation_counts = {
            ("validation", "stochastic"): validation_episodes,
            ("test", "stochastic"): stochastic_test_episodes,
            ("test", "greedy"): greedy_test_episodes,
        }
        evaluation_counts = evaluation.groupby(
            ["Variant", "Split", "Evaluation Mode"]
        ).size()
        for variant in expected_variants:
            for (split, mode), expected_count in expected_evaluation_counts.items():
                count = int(evaluation_counts.get((variant, split, mode), 0))
                if count != expected_count:
                    raise ValueError(
                        f"Seed {seed}/{variant}/{split}/{mode} has {count} "
                        f"episodes; expected {expected_count}"
                    )

        shard_roots[seed] = shard_root
        run_frames.append(runs)
        training_frames.append(training)
        evaluation_frames.append(evaluation)

    observed_seeds = sorted(shard_roots)
    if observed_seeds != expected_seeds:
        missing = sorted(set(expected_seeds) - set(observed_seeds))
        extra = sorted(set(observed_seeds) - set(expected_seeds))
        raise ValueError(
            f"Seed set is incomplete; missing={missing}, unexpected={extra}"
        )

    runs = pd.concat(run_frames, ignore_index=True)
    training = pd.concat(training_frames, ignore_index=True)
    evaluation = pd.concat(evaluation_frames, ignore_index=True)
    if runs.duplicated(["Seed", "Variant"]).any():
        raise ValueError("Merged run summaries contain duplicate seed/variant rows")
    return runs, training, evaluation, shard_roots


def _sort_by_variant(frame: pd.DataFrame, extra: list[str]) -> pd.DataFrame:
    order = {variant: index for index, variant in enumerate(VARIANTS)}
    result = frame.copy()
    result["_variant_order"] = result["Variant"].map(order)
    result = result.sort_values(["Seed", "_variant_order", *extra])
    return result.drop(columns="_variant_order").reset_index(drop=True)


def _verify_run_metrics(runs: pd.DataFrame, evaluation: pd.DataFrame) -> None:
    specifications = [
        (
            "validation",
            "stochastic",
            "Validation Reward Mean",
            "Validation Success Mean",
        ),
        ("test", "stochastic", "Test Reward Mean", "Test Success Mean"),
        (
            "test",
            "greedy",
            "Greedy Test Reward Mean",
            "Greedy Test Success Mean",
        ),
    ]
    for _, row in runs.iterrows():
        seed = int(row["Seed"])
        variant = row["Variant"]
        subset = evaluation[
            (evaluation["Seed"] == seed) & (evaluation["Variant"] == variant)
        ]
        for split, mode, reward_column, success_column in specifications:
            episodes = subset[
                (subset["Split"] == split)
                & (subset["Evaluation Mode"] == mode)
            ]
            observed_reward = float(episodes["Reward"].mean())
            observed_success = float(episodes["Success"].mean())
            expected_reward = float(row[reward_column])
            expected_success = float(row[success_column])
            if not np.isclose(observed_reward, expected_reward):
                raise ValueError(
                    f"{seed}/{variant} {reward_column} does not match raw episodes"
                )
            if not np.isclose(observed_success, expected_success):
                raise ValueError(
                    f"{seed}/{variant} {success_column} does not match raw episodes"
                )


def _copy_identical_result(
    shard_roots: dict[int, Path],
    filename: str,
    destination: Path,
) -> None:
    sources = [root / "results" / filename for root in shard_roots.values()]
    if any(not source.exists() for source in sources):
        raise FileNotFoundError(f"At least one shard is missing {filename}")
    digests = {_sha256(source) for source in sources}
    if len(digests) != 1:
        raise ValueError(f"Shard copies of {filename} are not identical")
    shutil.copy2(sources[0], destination)


def _copy_named_files(
    sources: list[Path],
    expected_names: set[str],
    destination: Path,
    label: str,
) -> None:
    by_name: dict[str, list[Path]] = {}
    for source in sources:
        by_name.setdefault(source.name, []).append(source)
    observed = set(by_name)
    if observed != expected_names:
        missing = sorted(expected_names - observed)
        extra = sorted(observed - expected_names)
        raise ValueError(f"{label} mismatch; missing={missing}, unexpected={extra}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, candidates in by_name.items():
        if not _files_identical(candidates):
            raise ValueError(f"Conflicting copies found for {label} {name}")
        shutil.copy2(candidates[0], destination / name)


def _copy_supporting_artifacts(
    shards_dir: Path,
    shard_roots: dict[int, Path],
    expected_seeds: list[int],
    expected_variants: list[str],
    output_dir: Path,
    map_counts: dict[str, int],
    require_videos: bool,
) -> dict[str, int]:
    checkpoint_sources = [
        path
        for root in shard_roots.values()
        for path in (root / "checkpoints").glob("*.pt")
    ]
    expected_checkpoints = {
        f"{variant}_{seed}.pt"
        for seed in expected_seeds
        for variant in expected_variants
    }
    _copy_named_files(
        checkpoint_sources,
        expected_checkpoints,
        output_dir / "checkpoints",
        "checkpoint",
    )

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for seed, shard_root in shard_roots.items():
        source = shard_root / "run.log"
        if not source.exists():
            raise FileNotFoundError(f"Seed {seed} is missing run.log")
        shutil.copy2(source, log_dir / f"seed_{seed:02d}.log")

    map_total = 0
    for split, expected_count in map_counts.items():
        map_sources = [
            path
            for path in shards_dir.rglob("map_*.npz")
            if path.parent.name == split
        ]
        expected_names = {f"map_{index:02d}.npz" for index in range(expected_count)}
        _copy_named_files(
            map_sources,
            expected_names,
            output_dir / "maps" / split,
            f"{split} map",
        )
        map_total += expected_count

    video_sources = list(shards_dir.rglob("*.gif"))
    expected_videos = {f"{variant}_0.gif" for variant in expected_variants}
    if require_videos:
        _copy_named_files(
            video_sources,
            expected_videos,
            output_dir / "videos",
            "video",
        )
    else:
        (output_dir / "videos").mkdir(parents=True, exist_ok=True)
        for source in video_sources:
            shutil.copy2(source, output_dir / "videos" / source.name)

    return {
        "checkpoints": len(expected_checkpoints),
        "logs": len(shard_roots),
        "maps": map_total,
        "videos": len(video_sources),
    }


def _write_summaries(runs: pd.DataFrame, args: argparse.Namespace) -> None:
    specifications = [
        (
            "training",
            "Train Reward Mean",
            "Curriculum Success Mean",
            "Train",
            True,
        ),
        (
            "validation",
            "Validation Reward Mean",
            "Validation Success Mean",
            "Validation",
            False,
        ),
        ("test", "Test Reward Mean", "Test Success Mean", "Test", False),
        (
            "greedy_test",
            "Greedy Test Reward Mean",
            "Greedy Test Success Mean",
            "Greedy Test",
            False,
        ),
    ]
    for name, reward, success, prefix, include_training in specifications:
        summary = summarize_runs(
            runs,
            reward_metric=reward,
            success_metric=success,
            stat_test=args.stat_test,
            prefix=prefix,
            include_training_details=include_training,
        )
        summary.to_csv(
            Path(args.results_dir) / f"{name}_summary.csv",
            index=False,
        )
        generate_results_table(
            summary,
            str(Path(args.results_dir) / f"{name}_results.html"),
        )


def _write_confirmatory_statistics(
    runs: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    comparators = [
        variant for variant in args.variants if variant != args.primary_variant
    ]
    reward_tests = confirmatory_comparisons(
        runs,
        metrics=["Test Reward Mean"],
        hybrid=args.primary_variant,
        comparators=comparators,
        minimum_effect=args.minimum_reward_effect,
    )
    success_tests = confirmatory_comparisons(
        runs,
        metrics=["Test Success Mean"],
        hybrid=args.primary_variant,
        comparators=comparators,
        minimum_effect=args.minimum_success_effect,
    )
    pd.concat([reward_tests, success_tests], ignore_index=True).to_csv(
        Path(args.results_dir) / "confirmatory_tests.csv",
        index=False,
    )
    pd.concat(
        [
            factorial_interaction(runs, "Test Reward Mean"),
            factorial_interaction(runs, "Test Success Mean"),
        ],
        ignore_index=True,
    ).to_csv(Path(args.results_dir) / "interaction_tests.csv", index=False)
    return write_confirmatory_decision(success_tests, args, comparators)


def _write_plots(training: pd.DataFrame, args: argparse.Namespace) -> None:
    for variant, specification in VARIANTS.items():
        variant_rows = training[training["Variant"] == variant]
        reward_logs: list[list[float]] = []
        intrinsic_logs: list[list[float]] = []
        success_logs: list[list[int]] = []
        for _, seed_rows in variant_rows.groupby("Seed", sort=True):
            seed_rows = seed_rows.sort_values("Episode")
            reward_logs.append(seed_rows["Environment Reward"].tolist())
            success_logs.append(seed_rows["Curriculum Success"].astype(int).tolist())
            if specification["intrinsic"]:
                intrinsic_logs.append(seed_rows["Intrinsic Reward"].tolist())
        plot_training_curves(
            reward_logs,
            intrinsic_logs if specification["intrinsic"] else None,
            success_logs,
            output_path=str(Path(args.plot_dir) / f"{variant}.pdf"),
        )
        plt.close("all")


def merge_confirmatory_shards(
    config_path: Path,
    shards_dir: Path,
    output_dir: Path,
    require_videos: bool = False,
) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    plots_dir = output_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    training_args = parse_training_args(
        [
            "--config",
            str(config_path),
            "--results-dir",
            str(results_dir),
            "--checkpoints-dir",
            str(output_dir / "checkpoints"),
            "--videos-dir",
            str(output_dir / "videos"),
            "--plot-dir",
            str(plots_dir),
            "--train-map-dir",
            str(output_dir / "maps" / "train"),
            "--validation-map-dir",
            str(output_dir / "maps" / "validation"),
            "--benchmark-map-dir",
            str(output_dir / "maps" / "test"),
        ]
    )
    expected_seeds = resolve_seeds(training_args)
    expected_variants = list(training_args.variants)
    runs, training, evaluation, shard_roots = _load_shards(
        shards_dir,
        expected_seeds,
        expected_variants,
        training_args.num_episodes,
        training_args.num_validation_maps * training_args.evaluation_rollouts,
        training_args.num_benchmark_maps * training_args.evaluation_rollouts,
        training_args.num_benchmark_maps
        * training_args.greedy_evaluation_rollouts,
    )
    _verify_run_metrics(runs, evaluation)

    runs = _sort_by_variant(runs, [])
    training = _sort_by_variant(training, ["Episode"])
    evaluation = _sort_by_variant(
        evaluation,
        ["Split", "Evaluation Mode", "Map", "Rollout"],
    )
    runs.to_csv(results_dir / "training_runs.csv", index=False)
    training.to_csv(results_dir / "training_episodes.csv", index=False)
    evaluation.to_csv(results_dir / "evaluation_episodes.csv", index=False)
    write_run_view(runs, training_args, "validation_runs.csv", "Validation")
    write_run_view(runs, training_args, "test_runs.csv", "Test")
    write_run_view(runs, training_args, "greedy_test_runs.csv", "Greedy Test")

    _copy_identical_result(
        shard_roots,
        "map_validity_episodes.csv",
        results_dir / "map_validity_episodes.csv",
    )
    _copy_identical_result(
        shard_roots,
        "map_validity_summary.csv",
        results_dir / "map_validity_summary.csv",
    )
    artifact_counts = _copy_supporting_artifacts(
        shards_dir,
        shard_roots,
        expected_seeds,
        expected_variants,
        output_dir,
        {
            "train": training_args.num_train_maps,
            "validation": training_args.num_validation_maps,
            "test": training_args.num_benchmark_maps,
        },
        require_videos,
    )

    write_protocol_manifest(training_args, expected_seeds)
    _write_summaries(runs, training_args)
    decision = _write_confirmatory_statistics(runs, training_args)
    calibration = write_calibration_decision(runs, training_args)
    _write_plots(training, training_args)

    inventory = {
        "status": "complete",
        "protocol": "confirmatory-pure-survival-v2",
        "seeds": expected_seeds,
        "variants": expected_variants,
        "trained_policies": len(runs),
        "training_episode_rows": len(training),
        "evaluation_episode_rows": len(evaluation),
        "primary_claim_supported": bool(decision["primary_claim_supported"]),
        "calibration_floor_met": bool(calibration["calibration_passed"]),
        **artifact_counts,
    }
    with (results_dir / "result_inventory.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(inventory, handle, sort_keys=False)
    return inventory


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    inventory = merge_confirmatory_shards(
        Path(args.config),
        Path(args.shards_dir),
        Path(args.output_dir),
        require_videos=args.require_videos,
    )
    print(yaml.safe_dump(inventory, sort_keys=False))


if __name__ == "__main__":
    main()
