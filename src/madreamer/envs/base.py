from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

ObservationDict = dict[str, np.ndarray]
ActionDict = dict[str, int]
RewardDict = dict[str, float]
DoneDict = dict[str, bool]
AliveDict = dict[str, bool]
InfoDict = dict[str, dict[str, object]]
EventDict = dict[str, dict[str, float]]


@dataclass
class StepResult:
    observations: ObservationDict
    rewards: RewardDict
    raw_rewards: RewardDict
    terminated: DoneDict
    truncated: DoneDict
    alive: AliveDict
    infos: InfoDict
    events: EventDict

    @property
    def done(self) -> bool:
        return all(self.terminated[agent_id] or self.truncated[agent_id] for agent_id in self.terminated)


class MultiAgentEnv(Protocol):
    agent_ids: tuple[str, ...]
    observation_shape: tuple[int, ...]
    action_dim: int
    last_infos: InfoDict

    def reset(self, seed: int | None = None) -> ObservationDict:
        ...

    def step(self, actions: ActionDict) -> StepResult:
        ...

    def close(self) -> None:
        ...
