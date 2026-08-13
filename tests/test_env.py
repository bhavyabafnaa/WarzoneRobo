import numpy as np
import torch

from src.env import (
    GridWorldICM,
    _observation_safe_action,
    evaluate_on_benchmarks,
    evaluate_trivial_policies,
    observable_action_mask,
)
from src.ppo import PPOPolicy


def test_step_boundaries_and_rewards():
    env = GridWorldICM(grid_size=3, max_steps=10)
    env.reset()
    env.cost_map = np.zeros((3, 3))
    env.risk_map = np.zeros((3, 3))
    env.cost_map[0, 1] = 0.6
    env.risk_map[0, 1] = 0.4

    # attempt move outside grid
    env.step(0)
    assert env.agent_pos == [0, 0]

    # move right into cost/risk cell
    _, reward, _, _, _ = env.step(3)
    assert env.agent_pos == [0, 1]
    assert reward <= -0.7


def test_dynamic_risk_updates_near_enemies():
    env = GridWorldICM(grid_size=3, dynamic_risk=True, max_steps=5, seed=0)
    env.reset(seed=0)
    env.risk_map = np.zeros((3, 3))
    env.enemy_positions = [[1, 1]]
    env.step(0)
    ex, ey = env.enemy_positions[0]
    assert env.risk_map[ex, ey] > 0


def test_dynamic_cost_updates_near_mines_and_decay():
    env = GridWorldICM(grid_size=3, dynamic_cost=True, max_steps=5, seed=0)
    env.reset(seed=0)
    env.cost_map = np.zeros((3, 3))
    env.cost_map[0, 0] = 1.0
    env.mine_map = np.zeros((3, 3), dtype=bool)
    env.mine_map[1, 1] = True
    env.step(1)  # move down, avoiding the mine
    # cost at previous high value should decay
    assert env.cost_map[0, 0] < 1.0
    # cost near the mine should increase
    assert env.cost_map[1, 1] > 0


def test_survival_reward_positive():
    env = GridWorldICM(grid_size=2, max_steps=5, survival_reward=0.05)
    env.reset()
    env.cost_map = np.zeros((2, 2))
    env.risk_map = np.zeros((2, 2))
    env.mine_map = np.zeros((2, 2), dtype=bool)
    env.enemy_positions = []
    _, reward, _, _, _ = env.step(1)
    assert reward > 0


def test_benchmark_evaluation_is_deterministic_and_reports_success(tmp_path):
    map_folder = tmp_path / "test_maps"
    map_folder.mkdir()
    env = GridWorldICM(grid_size=3, max_steps=4, seed=0)
    for index in range(2):
        env.reset(seed=10 + index)
        env.save_map(map_folder / f"map_{index:02d}.npz")

    policy = PPOPolicy(env.observation_size, 4)
    first = evaluate_on_benchmarks(
        env,
        policy,
        str(map_folder),
        num_maps=2,
        seed=500,
        deterministic=True,
        return_success=True,
    )
    second = evaluate_on_benchmarks(
        env,
        policy,
        str(map_folder),
        num_maps=2,
        seed=500,
        deterministic=True,
        return_success=True,
    )

    assert first == second
    assert len(first) == 4


def test_random_starts_are_seeded_safe_and_not_fixed_to_corner():
    env = GridWorldICM(
        grid_size=5,
        random_start=True,
        boundary_mines=True,
    )
    first_positions = []
    for seed in range(8):
        _, info = env.reset(seed=seed)
        position = info["start_position"]
        first_positions.append(position)
        assert not env.mine_map[position]
        assert list(position) not in env.enemy_positions
        assert 0 < position[0] < env.grid_size - 1
        assert 0 < position[1] < env.grid_size - 1

    _, repeated = env.reset(seed=3)
    assert repeated["start_position"] == first_positions[3]
    assert len(set(first_positions)) > 1


def test_enemy_positions_are_policy_observable():
    env = GridWorldICM(grid_size=5, seed=0)
    observation, _ = env.reset(seed=0)
    layers = observation.reshape(
        env.observation_channels,
        env.grid_size,
        env.grid_size,
    )

    assert layers.shape[0] == 5
    for enemy_row, enemy_column in env.enemy_positions:
        assert layers[4, enemy_row, enemy_column] == 1.0


def test_random_starts_respect_clearance_constraints():
    env = GridWorldICM(
        grid_size=8,
        random_start=True,
        boundary_mines=True,
        start_margin=2,
        start_mine_clearance=1,
        start_enemy_clearance=1,
    )
    for seed in range(12):
        _, info = env.reset(seed=seed)
        row, column = info["start_position"]
        assert min(row, column, 7 - row, 7 - column) >= 2
        assert all(
            abs(row - mine_row) + abs(column - mine_column) > 1
            for mine_row, mine_column in np.argwhere(env.mine_map)
        )
        assert all(
            abs(row - enemy_row) + abs(column - enemy_column) > 1
            for enemy_row, enemy_column in env.enemy_positions
        )


def test_safe_heuristic_uses_only_observable_hazards():
    grid_size = 5
    layers = np.zeros((5, grid_size, grid_size), dtype=np.float32)
    layers[0, 2, 2] = 1.0
    layers[3, 1, 2] = 1.0
    layers[4, 3, 2] = 1.0
    action = _observation_safe_action(
        layers.flatten(),
        grid_size,
        np.zeros((grid_size, grid_size), dtype=int),
        np.random.default_rng(0),
    )

    assert action in {2, 3}


def test_action_mask_excludes_observable_lethal_cells():
    layers = np.zeros((5, 5, 5), dtype=np.float32)
    layers[0, 2, 2] = 1.0
    layers[3, 1, 2] = 1.0
    layers[4, 3, 2] = 1.0

    mask = observable_action_mask(layers.flatten(), 5)

    assert mask.tolist() == [False, False, True, True]


def test_benchmark_policy_cannot_select_a_visible_lethal_action(tmp_path):
    map_folder = tmp_path / "masked_test_maps"
    map_folder.mkdir()
    env = GridWorldICM(grid_size=3, max_steps=1, seed=0)
    env.cost_map = np.zeros((3, 3))
    env.risk_map = np.zeros((3, 3))
    env.terrain_map = np.full((3, 3), "normal")
    env.mine_map = np.zeros((3, 3), dtype=bool)
    env.mine_map[0, 1] = True
    env.enemy_positions = []
    env.save_map(map_folder / "map_00.npz")
    policy = PPOPolicy(env.observation_size, 4)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        policy.actor.bias[3] = 100.0  # right is a visible mine
        policy.actor.bias[1] = 1.0  # down is the only valid move

    result = evaluate_on_benchmarks(
        env,
        policy,
        str(map_folder),
        num_maps=1,
        deterministic=True,
        return_success=True,
    )

    assert result[2] == 1.0


def test_survival_bonus_aligns_reward_with_survival():
    env = GridWorldICM(
        grid_size=3,
        max_steps=1,
        survival_reward=0.5,
        terminal_survival_bonus=5.0,
    )
    env.cost_map = np.zeros((3, 3))
    env.risk_map = np.zeros((3, 3))
    env.mine_map = np.zeros((3, 3), dtype=bool)
    env.terrain_map = np.full((3, 3), "normal")
    env.enemy_positions = []

    _, reward, done, _, info = env.step(1)

    assert done is True
    assert info["survived"] is True
    assert np.isclose(reward, 5.5)


def test_repeated_stochastic_evaluation_is_reproducible(tmp_path):
    map_folder = tmp_path / "test_maps"
    map_folder.mkdir()
    env = GridWorldICM(grid_size=4, max_steps=3, random_start=True, seed=0)
    for index in range(2):
        env.reset(seed=20 + index)
        env.save_map(map_folder / f"map_{index:02d}.npz")

    policy = PPOPolicy(env.observation_size, 4)
    first = evaluate_on_benchmarks(
        env,
        policy,
        str(map_folder),
        num_maps=2,
        seed=700,
        deterministic=False,
        return_success=True,
        num_rollouts=3,
        return_records=True,
    )
    second = evaluate_on_benchmarks(
        env,
        policy,
        str(map_folder),
        num_maps=2,
        seed=700,
        deterministic=False,
        return_success=True,
        num_rollouts=3,
        return_records=True,
    )

    assert first == second
    assert len(first[1]) == 6
    assert {record["Evaluation Mode"] for record in first[1]} == {"stochastic"}


def test_boundary_hazards_defeat_constant_action_policies(tmp_path):
    map_folder = tmp_path / "maps"
    map_folder.mkdir()
    env = GridWorldICM(
        grid_size=4,
        max_steps=4,
        random_start=True,
        boundary_mines=True,
    )
    env.reset(seed=0)
    env.save_map(map_folder / "map_00.npz")

    records = evaluate_trivial_policies(
        env,
        str(map_folder),
        num_maps=1,
        num_rollouts=3,
        seed=100,
        split="test",
    )
    constant_records = [
        record for record in records if record["Policy"].startswith("constant_")
    ]

    assert len(constant_records) == 12
    assert all(record["Success"] == 0 for record in constant_records)
    assert {record["Policy"] for record in records} >= {
        "masked_random",
        "masked_constant_0",
        "masked_constant_1",
        "masked_constant_2",
        "masked_constant_3",
    }
