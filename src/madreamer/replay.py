from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from random import sample

import numpy as np


@dataclass
class Transition:
    observations: dict[str, np.ndarray]
    actions: dict[str, int]
    rewards: dict[str, float]
    next_observations: dict[str, np.ndarray]
    terminated: dict[str, bool]
    truncated: dict[str, bool]
    infos: dict[str, dict[str, object]] | None = None


class MultiAgentReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._storage: deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self._storage.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size > len(self._storage):
            raise ValueError("Batch size exceeds number of stored transitions.")
        return sample(list(self._storage), batch_size)

    def __len__(self) -> int:
        return len(self._storage)
