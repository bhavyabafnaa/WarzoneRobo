import random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldModel(nn.Module):
    """Simple dynamics model f_psi predicting next state from (s, a)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, state_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict next state given current state and one-hot action."""
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ReplayBuffer:
    """Fixed-size buffer to store environment transitions."""

    def __init__(self, capacity: int = 10000):
        self.buffer: Deque[Tuple[np.ndarray, int, np.ndarray]] = deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: int, next_state: np.ndarray) -> None:
        self.buffer.append((state, action, next_state))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, next_states = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(next_states, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)
