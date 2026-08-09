import argparse
import os
from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import f_oneway, mannwhitneyu, ttest_ind, ttest_rel
from torch import optim

from src.env import GridWorldICM, evaluate_on_benchmarks, export_benchmark_maps
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
    parser.add_argument("--num_benchmark_maps", type=int, default=5)
    parser.add_argument("--evaluation_seed", type=int, default=10_000)
    parser.add_argument(
        "--log_backend",
        choices=["wandb", "tensorboard", "none"],
        default="none",
    )
    parser.add_argument("--plot-dir", default=None)
    parser.add_argument("--render-videos", action="store_true")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--videos-dir", default="videos")
    parser.add_argument("--training-map-dir", default="maps")
    parser.add_argument("--train-map-dir", default="train_maps")
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
    training_map = os.path.join(
        args.training_map_dir,
        f"train_seed_{run_seed}.npz",
    )

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
        training_map_path=training_map,
    )
    (
        rewards,
        intrinsic,
        _,
        _,
        paths,
        _,
        success,
        planner_usage,
    ) = metrics

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

    evaluation_seed = args.evaluation_seed + run_seed * args.num_benchmark_maps
    (
        benchmark_reward,
        benchmark_reward_std,
        benchmark_success,
        benchmark_success_std,
    ) = evaluate_on_benchmarks(
        env,
        policy,
        args.benchmark_map_dir,
        args.num_benchmark_maps,
        seed=evaluation_seed,
        deterministic=True,
        return_success=True,
    )

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
        "Train Reward Mean": float(np.mean(rewards)),
        "Train Reward Episode Std": sample_std(rewards),
        "Final Reward Mean": float(np.mean(rewards[-final_window:])),
        "Success Mean": float(np.mean(success)),
        "Final Success Mean": float(np.mean(success[-final_window:])),
        "Planner Consultation %": float(np.mean(planner_usage)) * 100,
        "Intrinsic Spikes": count_intrinsic_spikes(intrinsic),
        "Benchmark Reward Mean": benchmark_reward,
        "Benchmark Reward Map Std": benchmark_reward_std,
        "Benchmark Success Mean": benchmark_success,
        "Benchmark Success Map Std": benchmark_success_std,
    }
    curves = {
        "reward": rewards,
        "intrinsic": intrinsic if spec["intrinsic"] else None,
        "success": success,
        "paths": paths,
    }
    return record, curves


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
    benchmark: bool = False,
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

        prefix = "Benchmark" if benchmark else "Train"
        row = {
            "Model": VARIANTS[variant_key]["label"],
            f"{prefix} Reward Mean": float(group[reward_metric].mean()),
            f"{prefix} Reward Std": sample_std(group[reward_metric]),
            f"{prefix} Success Mean": float(group[success_metric].mean()),
            f"{prefix} Success Std": sample_std(group[success_metric]),
        }
        if not benchmark:
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


def main(argv=None):
    args = parse_args(argv)
    seeds = resolve_seeds(args)
    for directory in [
        args.results_dir,
        args.checkpoints_dir,
        args.videos_dir,
        args.training_map_dir,
        args.train_map_dir,
        args.benchmark_map_dir,
    ]:
        os.makedirs(directory, exist_ok=True)
    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)

    # Held-out maps are generated once and reused by every variant.  Their
    # seeds begin after the training-map seed range used by this helper.
    map_env = build_environment(args, seed=0)
    export_benchmark_maps(
        map_env,
        num_train=args.num_train_maps,
        num_test=args.num_benchmark_maps,
        train_folder=args.train_map_dir,
        test_folder=args.benchmark_map_dir,
    )

    logger = make_logger(args)
    run_records = []
    curve_logs = {
        key: {"reward": [], "intrinsic": [], "success": []} for key in args.variants
    }
    try:
        for run_seed in seeds:
            for variant_key in args.variants:
                record, curves = run_variant(
                    args,
                    variant_key,
                    run_seed,
                    logger=logger,
                )
                run_records.append(record)
                curve_logs[variant_key]["reward"].append(curves["reward"])
                if curves["intrinsic"] is not None:
                    curve_logs[variant_key]["intrinsic"].append(curves["intrinsic"])
                curve_logs[variant_key]["success"].append(curves["success"])
    finally:
        close_logger(logger, args.log_backend)

    runs = pd.DataFrame(run_records)
    runs.to_csv(os.path.join(args.results_dir, "training_runs.csv"), index=False)
    benchmark_columns = [
        "Seed",
        "Variant",
        "Model",
        "Benchmark Reward Mean",
        "Benchmark Reward Map Std",
        "Benchmark Success Mean",
        "Benchmark Success Map Std",
    ]
    runs[benchmark_columns].to_csv(
        os.path.join(args.results_dir, "benchmark_runs.csv"),
        index=False,
    )

    training_summary = summarize_runs(
        runs,
        reward_metric="Train Reward Mean",
        success_metric="Success Mean",
        stat_test=args.stat_test,
    )
    benchmark_summary = summarize_runs(
        runs,
        reward_metric="Benchmark Reward Mean",
        success_metric="Benchmark Success Mean",
        stat_test=args.stat_test,
        benchmark=True,
    )
    generate_results_table(
        training_summary,
        os.path.join(args.results_dir, "training_results.html"),
    )
    generate_results_table(
        benchmark_summary,
        os.path.join(args.results_dir, "benchmark_results.html"),
    )

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
