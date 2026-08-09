import argparse
import os
import sys
from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import f_oneway, mannwhitneyu, ttest_ind, ttest_rel
from torch import optim

from src.analysis import (
    confirmatory_comparisons,
    factorial_interaction,
    mean_confidence_interval,
)
from src.env import (
    GridWorldICM,
    evaluate_on_benchmarks,
    evaluate_trivial_policies,
    export_benchmark_maps,
)
from src.icm import ICMModule
from src.planner import SymbolicPlanner
from src.ppo import PPOPolicy, train_agent
from src.pseudocount import PseudoCountExploration
from src.rnd import RNDModule
from src.utils import count_intrinsic_spikes, save_model
from src.visualization import (
    generate_results_table,
    plot_training_curves,
    render_episode_video,
)

# This is the single experiment matrix used by training, tables, figures and
# the paper.  In particular, Planner-only is a real run rather than an implied
# ablation, and the duplicate episodic-count baseline has been removed.
VARIANTS: dict[str, dict] = {
    "ppo": {
        "label": "PPO Only",
        "intrinsic": False,
        "planner": False,
    },
    "icm": {
        "label": "PPO + ICM",
        "intrinsic": "icm",
        "planner": False,
    },
    "rnd": {
        "label": "PPO + RND",
        "intrinsic": "rnd",
        "planner": False,
    },
    "pc": {
        "label": "PPO + PC",
        "intrinsic": "pseudo",
        "planner": False,
    },
    "planner": {
        "label": "PPO + Planner",
        "intrinsic": False,
        "planner": True,
    },
    "icm_planner": {
        "label": "PPO + ICM + Planner",
        "intrinsic": "icm",
        "planner": True,
    },
}


def parse_args(argv=None):
    """Parse command-line arguments, with YAML values as defaults."""
    parser = argparse.ArgumentParser(description="Run WarzoneRobo experiments")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--grid_size", type=int, default=12)
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--survival_reward", type=float, default=0.05)
    parser.add_argument("--terminal_survival_bonus", type=float, default=0.0)
    parser.add_argument("--cost_weight", type=float, default=2.0)
    parser.add_argument("--risk_weight", type=float, default=3.0)
    parser.add_argument("--revisit_penalty", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--intrinsic_learning_rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--initial-beta", dest="initial_beta", type=float, default=0.1)
    parser.add_argument("--final-beta", dest="final_beta", type=float, default=None)
    parser.add_argument("--initial_planner_bonus", type=float, default=0.5)
    parser.add_argument("--final_planner_bonus", type=float, default=0.1)
    parser.add_argument("--dynamic_risk", action="store_true")
    parser.add_argument("--dynamic_cost", action="store_true")
    parser.add_argument("--add_noise", action="store_true")
    parser.add_argument("--random_start", action="store_true")
    parser.add_argument("--boundary_mines", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit run seeds, for example: --seeds 0 1 2 3 4",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help="Use seeds 0 through N-1 instead of an explicit seed list",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS),
        default=list(VARIANTS),
    )
    parser.add_argument("--num_train_maps", type=int, default=15)
    parser.add_argument("--num_validation_maps", type=int, default=5)
    parser.add_argument("--num_benchmark_maps", type=int, default=5)
    parser.add_argument("--evaluation_seed", type=int, default=10_000)
    parser.add_argument("--evaluation_rollouts", type=int, default=1)
    parser.add_argument("--greedy_evaluation_rollouts", type=int, default=1)
    parser.add_argument("--trivial_policy_rollouts", type=int, default=5)
    parser.add_argument("--max_constant_success", type=float, default=1.0)
    parser.add_argument("--minimum_safe_heuristic_success", type=float, default=0.0)
    parser.add_argument("--minimum_reward_effect", type=float, default=0.0)
    parser.add_argument("--minimum_success_effect", type=float, default=0.0)
    parser.add_argument(
        "--primary_variant",
        choices=list(VARIANTS),
        default="icm_planner",
    )
    parser.add_argument(
        "--log_backend",
        choices=["wandb", "tensorboard", "none"],
        default="none",
    )
    parser.add_argument("--plot-dir", default=None)
    parser.add_argument("--render-videos", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--videos-dir", default="videos")
    parser.add_argument("--train-map-dir", default="train_maps")
    parser.add_argument("--validation-map-dir", default="validation_maps")
    parser.add_argument("--benchmark-map-dir", default="test_maps")
    parser.add_argument(
        "--stat-test",
        choices=["paired", "welch", "mannwhitney", "anova"],
        default="paired",
    )

    config_args, _ = parser.parse_known_args(argv)
    if config_args.config and os.path.exists(config_args.config):
        with open(config_args.config, "r") as handle:
            parser.set_defaults(**(yaml.safe_load(handle) or {}))
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_seeds(args) -> list[int]:
    if args.num_seeds is not None:
        if args.num_seeds < 1:
            raise ValueError("--num-seeds must be at least 1")
        return list(range(args.num_seeds))
    return list(args.seeds) if args.seeds else [args.seed]


def sample_std(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def make_logger(args):
    if args.log_backend == "tensorboard":
        from torch.utils.tensorboard import SummaryWriter

        os.makedirs("logs/tensorboard", exist_ok=True)
        return SummaryWriter("logs/tensorboard")
    if args.log_backend == "wandb":
        import wandb

        wandb.init(project="warzone", config=vars(args))
        return wandb
    return None


def close_logger(logger, backend: str) -> None:
    if logger is None:
        return
    if backend == "tensorboard":
        logger.close()
    elif backend == "wandb":
        logger.finish()


def build_environment(args, seed: int) -> GridWorldICM:
    return GridWorldICM(
        grid_size=args.grid_size,
        dynamic_risk=args.dynamic_risk,
        dynamic_cost=args.dynamic_cost,
        max_steps=args.max_steps,
        survival_reward=args.survival_reward,
        terminal_survival_bonus=args.terminal_survival_bonus,
        random_start=args.random_start,
        boundary_mines=args.boundary_mines,
        seed=seed,
    )


def run_variant(args, variant_key: str, run_seed: int, logger=None):
    spec = VARIANTS[variant_key]
    set_seed(run_seed)
    env = build_environment(args, run_seed)
    input_dim = 4 * args.grid_size * args.grid_size
    action_dim = 4

    # The policy initialization is identical across variants for each seed.
    policy = PPOPolicy(input_dim, action_dim)
    policy_optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)
    icm = ICMModule(input_dim, action_dim)
    rnd = None
    pseudo = None
    intrinsic_optimizer = None

    if spec["intrinsic"] == "icm":
        intrinsic_optimizer = optim.Adam(
            icm.parameters(),
            lr=args.intrinsic_learning_rate,
        )
        use_icm = True
    elif spec["intrinsic"] == "rnd":
        rnd = RNDModule(input_dim)
        intrinsic_optimizer = optim.Adam(
            rnd.predictor.parameters(),
            lr=args.intrinsic_learning_rate,
        )
        use_icm = "rnd"
    elif spec["intrinsic"] == "pseudo":
        pseudo = PseudoCountExploration()
        use_icm = "pseudo"
    else:
        use_icm = False

    # Module construction consumes random numbers.  Reset the action-sampling
    # stream so all variants begin with the same policy RNG for a given seed.
    torch.manual_seed(run_seed)

    planner = SymbolicPlanner(
        env.cost_map,
        env.risk_map,
        env.np_random,
        cost_weight=args.cost_weight,
        risk_weight=args.risk_weight,
        revisit_penalty=args.revisit_penalty,
    )
    planner_weights = {
        "cost_weight": args.cost_weight,
        "risk_weight": args.risk_weight,
        "revisit_penalty": args.revisit_penalty,
    }
    training_maps = [
        os.path.join(args.train_map_dir, f"map_{map_index:02d}.npz")
        for map_index in range(args.num_train_maps)
    ]

    episode_records = []
    print(f"Training {spec['label']} (seed={run_seed})")
    metrics = train_agent(
        env,
        policy,
        icm,
        planner,
        policy_optimizer,
        intrinsic_optimizer,
        use_icm=use_icm,
        use_planner=spec["planner"],
        num_episodes=args.num_episodes,
        beta=args.initial_beta,
        final_beta=args.final_beta,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        planner_weights=planner_weights,
        rnd=rnd,
        pseudo=pseudo,
        seed=run_seed,
        add_noise=args.add_noise,
        logger=logger,
        initial_bonus=args.initial_planner_bonus,
        final_planner_bonus=args.final_planner_bonus,
        reset_env=False,
        training_map_paths=training_maps,
        episode_records=episode_records,
    )
    (
        rewards,
        intrinsic,
        _,
        planner_consultations,
        paths,
        step_counts,
        success,
        planner_usage,
    ) = metrics

    for episode_record in episode_records:
        episode_record.update(
            {
                "Seed": run_seed,
                "Variant": variant_key,
                "Model": spec["label"],
            }
        )

    checkpoint_path = os.path.join(
        args.checkpoints_dir,
        f"{variant_key}_{run_seed}.pt",
    )
    save_model(
        policy,
        checkpoint_path,
        icm=icm if spec["intrinsic"] == "icm" else None,
        rnd=rnd,
    )

    evaluation_seed = args.evaluation_seed + run_seed * 10_000_000
    validation_summary, validation_records = evaluate_on_benchmarks(
        env,
        policy,
        args.validation_map_dir,
        args.num_validation_maps,
        seed=evaluation_seed,
        deterministic=False,
        return_success=True,
        num_rollouts=args.evaluation_rollouts,
        return_records=True,
        split="validation",
    )
    test_seed = evaluation_seed + 1_000_000
    test_summary, test_records = evaluate_on_benchmarks(
        env,
        policy,
        args.benchmark_map_dir,
        args.num_benchmark_maps,
        seed=test_seed,
        deterministic=False,
        return_success=True,
        num_rollouts=args.evaluation_rollouts,
        return_records=True,
        split="test",
    )
    greedy_test_summary, greedy_test_records = evaluate_on_benchmarks(
        env,
        policy,
        args.benchmark_map_dir,
        args.num_benchmark_maps,
        seed=test_seed,
        deterministic=True,
        return_success=True,
        num_rollouts=args.greedy_evaluation_rollouts,
        return_records=True,
        split="test",
    )
    evaluation_records = validation_records + test_records + greedy_test_records
    for evaluation_record in evaluation_records:
        evaluation_record.update(
            {
                "Seed": run_seed,
                "Variant": variant_key,
                "Model": spec["label"],
            }
        )
    (
        validation_reward,
        validation_reward_std,
        validation_success,
        validation_success_std,
    ) = validation_summary
    (
        test_reward,
        test_reward_std,
        test_success,
        test_success_std,
    ) = test_summary
    (
        greedy_test_reward,
        greedy_test_reward_std,
        greedy_test_success,
        greedy_test_success_std,
    ) = greedy_test_summary

    if args.render_videos:
        render_episode_video(
            env,
            policy,
            os.path.join(args.videos_dir, f"{variant_key}_{run_seed}.gif"),
            max_steps=args.max_steps,
            seed=args.evaluation_seed + run_seed,
        )

    final_window = min(50, len(rewards))
    record = {
        "Seed": run_seed,
        "Variant": variant_key,
        "Model": spec["label"],
        "Training Maps": args.num_train_maps,
        "Train Reward Mean": float(np.mean(rewards)),
        "Train Reward Episode Std": sample_std(rewards),
        "Final Reward Mean": float(np.mean(rewards[-final_window:])),
        "Success Mean": float(np.mean(success)),
        "Final Success Mean": float(np.mean(success[-final_window:])),
        "Planner Consultation %": float(np.mean(planner_usage)) * 100,
        "Intrinsic Spikes": count_intrinsic_spikes(intrinsic),
        "Validation Reward Mean": validation_reward,
        "Validation Reward Episode Std": validation_reward_std,
        "Validation Success Mean": validation_success,
        "Validation Success Episode Std": validation_success_std,
        "Test Reward Mean": test_reward,
        "Test Reward Episode Std": test_reward_std,
        "Test Success Mean": test_success,
        "Test Success Episode Std": test_success_std,
        "Greedy Test Reward Mean": greedy_test_reward,
        "Greedy Test Reward Episode Std": greedy_test_reward_std,
        "Greedy Test Success Mean": greedy_test_success,
        "Greedy Test Success Episode Std": greedy_test_success_std,
    }
    curves = {
        "reward": rewards,
        "intrinsic": intrinsic if spec["intrinsic"] else None,
        "success": success,
        "paths": paths,
        "steps": step_counts,
        "planner_consultations": planner_consultations,
    }
    return record, curves, episode_records, evaluation_records


def comparison_pvalue(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    metric: str,
    test: str,
) -> float:
    paired = baseline[["Seed", metric]].merge(
        candidate[["Seed", metric]],
        on="Seed",
        suffixes=("_base", "_candidate"),
    )
    base_values = paired[f"{metric}_base"].to_numpy()
    candidate_values = paired[f"{metric}_candidate"].to_numpy()
    if len(base_values) < 2:
        return float("nan")
    if np.allclose(base_values, candidate_values):
        return 1.0
    if test == "paired":
        return float(ttest_rel(base_values, candidate_values).pvalue)
    if test == "welch":
        return float(ttest_ind(base_values, candidate_values, equal_var=False).pvalue)
    if test == "mannwhitney":
        return float(
            mannwhitneyu(
                base_values,
                candidate_values,
                alternative="two-sided",
            ).pvalue
        )
    return float("nan")


def summarize_runs(
    runs: pd.DataFrame,
    reward_metric: str,
    success_metric: str,
    stat_test: str,
    prefix: str,
    include_training_details: bool = False,
) -> pd.DataFrame:
    baseline = runs[runs["Variant"] == "ppo"]
    reward_anova = float("nan")
    success_anova = float("nan")
    if stat_test == "anova":
        groups = [group for _, group in runs.groupby("Variant")]
        if len(groups) >= 3 and all(len(group) >= 2 for group in groups):
            reward_anova = float(
                f_oneway(*(group[reward_metric] for group in groups)).pvalue
            )
            success_anova = float(
                f_oneway(*(group[success_metric] for group in groups)).pvalue
            )

    rows = []
    for variant_key in VARIANTS:
        group = runs[runs["Variant"] == variant_key]
        if group.empty:
            continue
        if variant_key == "ppo":
            reward_p = float("nan")
            success_p = float("nan")
        elif stat_test == "anova":
            reward_p = reward_anova
            success_p = success_anova
        else:
            reward_p = comparison_pvalue(
                baseline,
                group,
                reward_metric,
                stat_test,
            )
            success_p = comparison_pvalue(
                baseline,
                group,
                success_metric,
                stat_test,
            )

        reward_mean, reward_std, reward_ci_low, reward_ci_high = (
            mean_confidence_interval(group[reward_metric])
        )
        success_mean, success_std, success_ci_low, success_ci_high = (
            mean_confidence_interval(group[success_metric])
        )
        row = {
            "Model": VARIANTS[variant_key]["label"],
            f"{prefix} Reward Mean": reward_mean,
            f"{prefix} Reward Std": reward_std,
            f"{prefix} Reward 95% CI Low": reward_ci_low,
            f"{prefix} Reward 95% CI High": reward_ci_high,
            f"{prefix} Success Mean": success_mean,
            f"{prefix} Success Std": success_std,
            f"{prefix} Success 95% CI Low": success_ci_low,
            f"{prefix} Success 95% CI High": success_ci_high,
        }
        if include_training_details:
            row.update(
                {
                    "Final Reward Mean": (float(group["Final Reward Mean"].mean())),
                    "Final Reward Std": sample_std(group["Final Reward Mean"]),
                    "Final Success Mean": (float(group["Final Success Mean"].mean())),
                    "Final Success Std": sample_std(group["Final Success Mean"]),
                    "Planner Consultation %": float(
                        group["Planner Consultation %"].mean()
                    ),
                }
            )
        row["Reward p-value"] = reward_p
        row["Success p-value"] = success_p
        rows.append(row)
    return pd.DataFrame(rows)


def write_protocol_manifest(args, seeds: list[int]) -> None:
    manifest = {
        "protocol": "confirmatory-pure-survival-v1",
        "python": sys.version.split()[0],
        "seeds": seeds,
        "declared_primary_comparison": (
            f"{args.primary_variant} versus every other selected variant"
        ),
        "primary_metric": "Test Success Mean",
        "secondary_metric": "Test Reward Mean",
        "multiple_comparison_correction": "Holm",
        "arguments": vars(args),
    }
    with open(
        os.path.join(args.results_dir, "protocol_manifest.yaml"),
        "w",
        encoding="utf-8",
    ) as handle:
        yaml.safe_dump(manifest, handle, sort_keys=True)


def validate_map_protocol(args, env: GridWorldICM) -> pd.DataFrame:
    split_specs = [
        ("train", args.train_map_dir, args.num_train_maps, 0),
        (
            "validation",
            args.validation_map_dir,
            args.num_validation_maps,
            10_000_000,
        ),
        ("test", args.benchmark_map_dir, args.num_benchmark_maps, 20_000_000),
    ]
    records = []
    for split, folder, num_maps, seed_offset in split_specs:
        records.extend(
            evaluate_trivial_policies(
                env,
                folder,
                num_maps,
                args.trivial_policy_rollouts,
                args.evaluation_seed + seed_offset,
                split,
            )
        )

    audit = pd.DataFrame(records)
    audit.to_csv(
        os.path.join(args.results_dir, "map_validity_episodes.csv"),
        index=False,
    )
    summary = (
        audit.groupby(["Split", "Policy"], as_index=False)
        .agg(
            Reward_Mean=("Reward", "mean"),
            Reward_Std=("Reward", "std"),
            Success_Mean=("Success", "mean"),
            Mean_Steps=("Steps", "mean"),
            Episodes=("Success", "size"),
        )
        .fillna({"Reward_Std": 0.0})
    )
    summary.to_csv(
        os.path.join(args.results_dir, "map_validity_summary.csv"),
        index=False,
    )

    constant_rows = summary[summary["Policy"].str.startswith("constant_")]
    failures = constant_rows[constant_rows["Success_Mean"] > args.max_constant_success]
    if not failures.empty:
        details = ", ".join(
            f"{row.Split}/{row.Policy}={row.Success_Mean:.3f}"
            for row in failures.itertuples()
        )
        raise RuntimeError(
            "Map validity gate failed; constant-action success exceeds "
            f"{args.max_constant_success:.3f}: {details}"
        )
    safe_rows = summary[summary["Policy"] == "safe_heuristic"]
    expected_splits = {split for split, *_ in split_specs}
    observed_safe_splits = set(safe_rows["Split"])
    if observed_safe_splits != expected_splits:
        missing = ", ".join(sorted(expected_splits - observed_safe_splits))
        raise RuntimeError(
            "Map solvability gate did not evaluate the safe heuristic on: " + missing
        )
    unsolved_splits = safe_rows[
        safe_rows["Success_Mean"] < args.minimum_safe_heuristic_success
    ]
    if not unsolved_splits.empty:
        details = ", ".join(
            f"{row.Split}={row.Success_Mean:.3f}"
            for row in unsolved_splits.itertuples()
        )
        raise RuntimeError(
            "Map solvability gate failed; safe-heuristic success is below "
            f"{args.minimum_safe_heuristic_success:.3f}: {details}"
        )
    return summary


def persist_progress(
    args,
    run_records: list[dict],
    episode_records: list[dict],
    evaluation_records: list[dict],
) -> None:
    pd.DataFrame(run_records).to_csv(
        os.path.join(args.results_dir, "training_runs.csv"),
        index=False,
    )
    pd.DataFrame(episode_records).to_csv(
        os.path.join(args.results_dir, "training_episodes.csv"),
        index=False,
    )
    pd.DataFrame(evaluation_records).to_csv(
        os.path.join(args.results_dir, "evaluation_episodes.csv"),
        index=False,
    )


def write_run_view(
    runs: pd.DataFrame,
    args,
    filename: str,
    metric_prefix: str,
) -> None:
    columns = [
        "Seed",
        "Variant",
        "Model",
        f"{metric_prefix} Reward Mean",
        f"{metric_prefix} Reward Episode Std",
        f"{metric_prefix} Success Mean",
        f"{metric_prefix} Success Episode Std",
    ]
    runs[columns].to_csv(
        os.path.join(args.results_dir, filename),
        index=False,
    )


def main(argv=None):
    args = parse_args(argv)
    seeds = resolve_seeds(args)
    count_arguments = {
        "--num_train_maps": args.num_train_maps,
        "--num_validation_maps": args.num_validation_maps,
        "--num_benchmark_maps": args.num_benchmark_maps,
        "--evaluation_rollouts": args.evaluation_rollouts,
        "--greedy_evaluation_rollouts": args.greedy_evaluation_rollouts,
        "--trivial_policy_rollouts": args.trivial_policy_rollouts,
    }
    for argument, value in count_arguments.items():
        if value < 1:
            raise ValueError(f"{argument} must be at least 1")
    if not 0 <= args.max_constant_success <= 1:
        raise ValueError("--max_constant_success must be between 0 and 1")
    if not 0 <= args.minimum_safe_heuristic_success <= 1:
        raise ValueError("--minimum_safe_heuristic_success must be between 0 and 1")
    if args.primary_variant not in args.variants:
        raise ValueError("--primary_variant must be included in --variants")

    for directory in [
        args.results_dir,
        args.checkpoints_dir,
        args.videos_dir,
        args.train_map_dir,
        args.validation_map_dir,
        args.benchmark_map_dir,
    ]:
        os.makedirs(directory, exist_ok=True)
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)

    write_protocol_manifest(args, seeds)
    map_env = build_environment(args, seed=0)
    export_benchmark_maps(
        map_env,
        num_train=args.num_train_maps,
        num_validation=args.num_validation_maps,
        num_test=args.num_benchmark_maps,
        train_folder=args.train_map_dir,
        validation_folder=args.validation_map_dir,
        test_folder=args.benchmark_map_dir,
    )
    validate_map_protocol(args, map_env)
    if args.validate_only:
        return

    logger = make_logger(args)
    run_records = []
    all_episode_records = []
    all_evaluation_records = []
    curve_logs = {
        key: {"reward": [], "intrinsic": [], "success": []} for key in args.variants
    }
    try:
        for run_seed in seeds:
            for variant_key in args.variants:
                record, curves, episode_records, evaluation_records = run_variant(
                    args,
                    variant_key,
                    run_seed,
                    logger=logger,
                )
                run_records.append(record)
                all_episode_records.extend(episode_records)
                all_evaluation_records.extend(evaluation_records)
                persist_progress(
                    args,
                    run_records,
                    all_episode_records,
                    all_evaluation_records,
                )
                curve_logs[variant_key]["reward"].append(curves["reward"])
                if curves["intrinsic"] is not None:
                    curve_logs[variant_key]["intrinsic"].append(curves["intrinsic"])
                curve_logs[variant_key]["success"].append(curves["success"])
    finally:
        close_logger(logger, args.log_backend)

    runs = pd.DataFrame(run_records)
    write_run_view(runs, args, "validation_runs.csv", "Validation")
    write_run_view(runs, args, "test_runs.csv", "Test")
    write_run_view(runs, args, "greedy_test_runs.csv", "Greedy Test")

    summaries = [
        (
            "training_results.html",
            summarize_runs(
                runs,
                reward_metric="Train Reward Mean",
                success_metric="Success Mean",
                stat_test=args.stat_test,
                prefix="Train",
                include_training_details=True,
            ),
        ),
        (
            "validation_results.html",
            summarize_runs(
                runs,
                reward_metric="Validation Reward Mean",
                success_metric="Validation Success Mean",
                stat_test=args.stat_test,
                prefix="Validation",
            ),
        ),
        (
            "test_results.html",
            summarize_runs(
                runs,
                reward_metric="Test Reward Mean",
                success_metric="Test Success Mean",
                stat_test=args.stat_test,
                prefix="Test",
            ),
        ),
        (
            "greedy_test_results.html",
            summarize_runs(
                runs,
                reward_metric="Greedy Test Reward Mean",
                success_metric="Greedy Test Success Mean",
                stat_test=args.stat_test,
                prefix="Greedy Test",
            ),
        ),
    ]
    for filename, summary in summaries:
        generate_results_table(summary, os.path.join(args.results_dir, filename))

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
    confirmatory_tests = pd.concat(
        [reward_tests, success_tests],
        ignore_index=True,
    )
    confirmatory_tests.to_csv(
        os.path.join(args.results_dir, "confirmatory_tests.csv"),
        index=False,
    )
    interaction_tests = pd.concat(
        [
            factorial_interaction(runs, "Test Reward Mean"),
            factorial_interaction(runs, "Test Success Mean"),
        ],
        ignore_index=True,
    )
    interaction_tests.to_csv(
        os.path.join(args.results_dir, "interaction_tests.csv"),
        index=False,
    )
    decision = {
        "primary_claim_supported": bool(
            not success_tests.empty and success_tests["Superiority Criterion Met"].all()
        ),
        "hybrid": args.primary_variant,
        "primary_metric": "Test Success Mean",
        "secondary_metric": "Test Reward Mean",
        "required_comparators": comparators,
        "minimum_mean_success_effect": args.minimum_success_effect,
        "secondary_minimum_mean_reward_effect": args.minimum_reward_effect,
        "holm_family_size": len(comparators),
    }
    with open(
        os.path.join(args.results_dir, "confirmatory_decision.yaml"),
        "w",
        encoding="utf-8",
    ) as handle:
        yaml.safe_dump(decision, handle, sort_keys=True)

    if args.plot_dir:
        for variant_key, logs in curve_logs.items():
            plot_training_curves(
                logs["reward"],
                logs["intrinsic"] or None,
                logs["success"],
                output_path=os.path.join(args.plot_dir, f"{variant_key}.pdf"),
            )


if __name__ == "__main__":
    main()
