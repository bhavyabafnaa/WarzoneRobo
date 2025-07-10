import numpy as np
from src.env import GridWorldICM


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
