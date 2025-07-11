import numpy as np


class SymbolicPlanner:
    """A simple scoring-based planner used to select safer actions."""

    def __init__(self, cost_map, risk_map, goal_pos, np_random,
                 cost_weight=2.0, risk_weight=3.0, goal_weight=0.5, revisit_penalty=1.0):
        self.cost_map = cost_map
        self.risk_map = risk_map
        self.goal_pos = goal_pos
        self.np_random = np_random
        self.cost_weight = cost_weight
        self.risk_weight = risk_weight
        self.goal_weight = goal_weight
        self.revisit_penalty = revisit_penalty
        self.grid_size = cost_map.shape[0]
        self.visited_map = np.zeros_like(cost_map, dtype=bool)

    def get_safe_subgoal(self, agent_pos):
        x, y = agent_pos
        directions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

        def score_func(pos):
            i, j = pos
            cost = self.cost_map[i][j]
            risk = self.risk_map[i][j]
            revisit = self.revisit_penalty if self.visited_map[i][j] else 0
            dist = np.linalg.norm(np.array([i, j]) - np.array(self.goal_pos))
            return (self.cost_weight * cost +
                    self.risk_weight * risk +
                    self.goal_weight * dist +
                    revisit)

        best_score = float('inf')
        best_action = None
        for action, (dx, dy) in directions.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                s = score_func((nx, ny))
                if s < best_score:
                    best_score = s
                    best_action = action

        self.visited_map[x][y] = True
        if best_action is None:
            return self.np_random.choice([0, 1, 2, 3])
        return best_action
