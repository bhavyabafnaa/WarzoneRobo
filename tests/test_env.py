import numpy as np

from src.env import GridWorldICM, evaluate_on_benchmarks
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
