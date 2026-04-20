from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

ObservationDict = dict[str, np.ndarray]
ActionDict = dict[str, int]
RewardDict = dict[str, float]
DoneDict = dict[str, bool]
InfoDict = dict[str, dict[str, object]]


@dataclass
class StepResult:
    observations: ObservationDict
    rewards: RewardDict
    terminated: DoneDict
    truncated: DoneDict
    infos: InfoDict

    @property
    def done(self) -> bool:
        return all(self.terminated.values()) or all(self.truncated.values())


class MultiAgentEnv(Protocol):
    agent_ids: tuple[str, ...]
    observation_shape: tuple[int, ...]
    action_dim: int

    def reset(self, seed: int | None = None) -> ObservationDict:
        ...

    def step(self, actions: ActionDict) -> StepResult:
        ...
