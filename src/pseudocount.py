import hashlib
from collections import defaultdict

import numpy as np


class PseudoCountExploration:
    """Simple hashed pseudo-count exploration bonus.

    States are hashed to a fixed bucket using their byte representation. The
    returned bonus is 1/sqrt(N) where N is the visitation count for the hashed
    state.
    """

    def __init__(self, hash_dim: int = 2**16):
        self.hash_dim = hash_dim
        self.counts = defaultdict(float)

    def _hash(self, state: np.ndarray) -> int:
        # Python's built-in hash is randomized between interpreter processes,
        # which made nominally seeded runs produce different pseudo-counts.
        # BLAKE2 gives a compact, stable bucket assignment across runs.
        state_bytes = np.ascontiguousarray(state).tobytes()
        digest = hashlib.blake2b(state_bytes, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little") % self.hash_dim

    def bonus(self, state: np.ndarray) -> float:
        idx = self._hash(state)
        self.counts[idx] += 1.0
        return 1.0 / np.sqrt(self.counts[idx])

    def reset(self):
        self.counts.clear()
