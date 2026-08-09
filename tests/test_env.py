import numpy as np

from src.env import (
    GridWorldICM,
    evaluate_on_benchmarks,
    evaluate_trivial_policies,
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

    policy = PPOPolicy(4 * env.grid_size * env.grid_size, 4)
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

    policy = PPOPolicy(4 * env.grid_size * env.grid_size, 4)
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
