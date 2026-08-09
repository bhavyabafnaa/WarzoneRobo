import numpy as np
import torch
import yaml
from torch import optim

from src.env import GridWorldICM
from src.icm import ICMModule
from src.planner import SymbolicPlanner
from src.ppo import PPOPolicy, train_agent


class RecordingPolicy(PPOPolicy):
    def __init__(self, input_dim, action_dim):
        super().__init__(input_dim, action_dim)
        self.forward_inputs = []

    def forward(self, x):
        self.forward_inputs.append(x.detach().clone())
        return super().forward(x)


class FixedActionPolicy(PPOPolicy):
    def __init__(self, input_dim, action_dim, fixed_action):
        super().__init__(input_dim, action_dim)
        self.fixed_action = fixed_action

    def act(self, x):
        logits, value = self.forward(x)
        distribution = torch.distributions.Categorical(logits=logits)
        action = torch.tensor(self.fixed_action)
        return self.fixed_action, distribution.log_prob(action), value


def test_short_training_loop(tmp_path):
    env = GridWorldICM(grid_size=4, max_steps=10)
    map_path = tmp_path / "training_map.npz"
    env.save_map(map_path)

    input_dim = 4 * env.grid_size * env.grid_size
    action_dim = 4
    policy = PPOPolicy(input_dim, action_dim)
    icm = ICMModule(input_dim, action_dim)
    planner = SymbolicPlanner(env.cost_map, env.risk_map, env.np_random)
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
        training_map_path=str(map_path),
    )


def test_training_one_episode_metrics(tmp_path):
    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    env = GridWorldICM(grid_size=cfg.get("grid_size", 4), max_steps=10)
    map_path = tmp_path / "training_map.npz"
    env.save_map(map_path)

    input_dim = 4 * env.grid_size * env.grid_size
    action_dim = 4
    policy = PPOPolicy(input_dim, action_dim)
    icm = ICMModule(input_dim, action_dim)
    planner = SymbolicPlanner(env.cost_map, env.risk_map, env.np_random)
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
        training_map_path=str(map_path),
    )

    rewards, _, _, _, _, _, _, _ = metrics
    assert len(rewards) == 1


def test_success_flag_survival(tmp_path):
    env = GridWorldICM(grid_size=2, max_steps=2)
    env.cost_map = np.zeros((2, 2))
    env.risk_map = np.zeros((2, 2))
    env.mine_map = np.zeros((2, 2), dtype=bool)
    env.enemy_positions = []
    map_path = tmp_path / "training_map.npz"
    env.save_map(map_path)

    input_dim = 4 * env.grid_size * env.grid_size
    action_dim = 4
    policy = PPOPolicy(input_dim, action_dim)
    icm = ICMModule(input_dim, action_dim)
    planner = SymbolicPlanner(env.cost_map, env.risk_map, env.np_random)
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
        reset_env=False,
        training_map_path=str(map_path),
    )

    _, _, _, _, _, _, success_flags, _ = metrics
    assert success_flags == [1]


def test_training_update_uses_state_where_action_was_selected(tmp_path):
    env = GridWorldICM(grid_size=2, max_steps=2, seed=0)
    env.cost_map = np.zeros((2, 2))
    env.risk_map = np.zeros((2, 2))
    env.mine_map = np.zeros((2, 2), dtype=bool)
    env.enemy_positions = []
    map_path = tmp_path / "training_map.npz"
    env.save_map(map_path)
    initial_obs, _ = env.reset(seed=0, load_map_path=map_path)

    policy = RecordingPolicy(4 * env.grid_size * env.grid_size, 4)
    icm = ICMModule(4 * env.grid_size * env.grid_size, 4)
    planner = SymbolicPlanner(env.cost_map, env.risk_map, env.np_random)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    train_agent(
        env,
        policy,
        icm,
        planner,
        optimizer,
        None,
        use_icm=False,
        use_planner=False,
        num_episodes=1,
        reset_env=False,
        training_map_path=str(map_path),
    )

    update_batch = [value for value in policy.forward_inputs if value.shape[0] == 2][-1]
    assert torch.allclose(update_batch[0], torch.tensor(initial_obs))


def make_high_risk_start_env(tmp_path):
    env = GridWorldICM(
        grid_size=2,
        max_steps=1,
        survival_reward=0.05,
        seed=0,
    )
    env.cost_map = np.zeros((2, 2))
    env.risk_map = np.zeros((2, 2))
    env.risk_map[0, 0] = 1.0
    env.mine_map = np.zeros((2, 2), dtype=bool)
    env.enemy_positions = []
    map_path = tmp_path / "high_risk_map.npz"
    env.save_map(map_path)
    return env, map_path


def test_planner_does_not_override_policy_action(tmp_path):
    env, map_path = make_high_risk_start_env(tmp_path)
    input_dim = 4 * env.grid_size * env.grid_size
    policy = FixedActionPolicy(input_dim, 4, fixed_action=0)
    icm = ICMModule(input_dim, 4)
    planner = SymbolicPlanner(env.cost_map, env.risk_map, env.np_random)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    metrics = train_agent(
        env,
        policy,
        icm,
        planner,
        optimizer,
        None,
        use_icm=False,
        use_planner=True,
        num_episodes=1,
        reset_env=False,
        training_map_path=str(map_path),
    )

    _, _, _, _, paths, _, _, planner_usage = metrics
    assert paths[0][-1] == (0, 0)
    assert planner_usage == [1.0]


def test_planner_bonus_is_not_reported_as_environment_reward(tmp_path):
    env, map_path = make_high_risk_start_env(tmp_path)

    input_dim = 4 * env.grid_size * env.grid_size
    policy = FixedActionPolicy(input_dim, 4, fixed_action=1)
    icm = ICMModule(input_dim, 4)
    planner = SymbolicPlanner(env.cost_map, env.risk_map, env.np_random)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    metrics = train_agent(
        env,
        policy,
        icm,
        planner,
        optimizer,
        None,
        use_icm=False,
        use_planner=True,
        num_episodes=1,
        initial_bonus=100.0,
        final_planner_bonus=100.0,
        reset_env=False,
        training_map_path=str(map_path),
    )

    rewards, _, _, _, _, _, _, planner_usage = metrics
    assert np.isclose(rewards[0], 0.05)
    assert planner_usage == [1.0]
