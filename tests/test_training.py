import os
import torch
from torch import optim

from src.env import GridWorldICM
from src.icm import ICMModule
from src.planner import SymbolicPlanner
from src.ppo import PPOPolicy, train_agent
import yaml


def test_short_training_loop(tmp_path):
    env = GridWorldICM(grid_size=4, max_steps=10)
    os.makedirs("maps", exist_ok=True)
    env.save_map("maps/map_00.npz")

    input_dim = 5 * env.grid_size * env.grid_size
    action_dim = 4
    policy = PPOPolicy(input_dim, action_dim)
    icm = ICMModule(input_dim, action_dim)
    planner = SymbolicPlanner(env.cost_map, env.risk_map, env.goal_pos, env.np_random)
    opt = optim.Adam(policy.parameters(), lr=1e-3)

    train_agent(
        env,
        policy,
        icm,
        planner,
        opt,
        opt,
        use_icm=False,
        use_planner=False,
        num_episodes=1,
    )


def test_training_one_episode_metrics(tmp_path):
    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    env = GridWorldICM(grid_size=cfg.get("grid_size", 4), max_steps=10)
    os.makedirs("maps", exist_ok=True)
    env.save_map("maps/map_00.npz")

    input_dim = 5 * env.grid_size * env.grid_size
    action_dim = 4
    policy = PPOPolicy(input_dim, action_dim)
    icm = ICMModule(input_dim, action_dim)
    planner = SymbolicPlanner(env.cost_map, env.risk_map, env.goal_pos, env.np_random)
    opt = optim.Adam(policy.parameters(), lr=1e-3)

    metrics = train_agent(
        env,
        policy,
        icm,
        planner,
        opt,
        opt,
        use_icm=False,
        use_planner=False,
        num_episodes=1,
    )

    rewards, _, _, _, _, _, success_flags, _ = metrics
    assert len(rewards) == 1
    assert len(success_flags) == 1
    assert all(flag in (0, 1) for flag in success_flags)
