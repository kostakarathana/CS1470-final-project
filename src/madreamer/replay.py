from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from random import sample

import numpy as np


@dataclass
class ReplayStep:
    episode_id: int
    observations: dict[str, np.ndarray]
    actions: dict[str, int]
    opponent_actions: dict[str, np.ndarray]
    rewards: dict[str, float]
    raw_rewards: dict[str, float]
    next_observations: dict[str, np.ndarray]
    terminated: dict[str, bool]
    truncated: dict[str, bool]
    alive: dict[str, bool]
    infos: dict[str, dict[str, object]]
    events: dict[str, dict[str, float]]


@dataclass
class AgentSequenceBatch:
    observations: np.ndarray
    actions: np.ndarray
    opponent_actions: np.ndarray
    rewards: np.ndarray
    raw_rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    continues: np.ndarray
    alive: np.ndarray


@dataclass
class SequenceBatch:
    episode_ids: np.ndarray
    agents: dict[str, AgentSequenceBatch]


class MultiAgentReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._storage: deque[ReplayStep] = deque(maxlen=capacity)

    def add(self, transition: ReplayStep) -> None:
        self._storage.append(transition)

    def sample(self, batch_size: int) -> list[ReplayStep]:
        if batch_size > len(self._storage):
            raise ValueError("Batch size exceeds number of stored transitions.")
        return sample(list(self._storage), batch_size)

    def sample_sequences(
        self,
        batch_size: int,
        sequence_length: int,
        agent_ids: tuple[str, ...],
    ) -> SequenceBatch:
        valid_starts = self._valid_sequence_starts(sequence_length)
        if batch_size > len(valid_starts):
            raise ValueError("Batch size exceeds number of valid replay sequences.")
        starts = sample(valid_starts, batch_size)
        sequences = [[self._storage[start + offset] for offset in range(sequence_length)] for start in starts]
        episode_ids = np.array([sequence[0].episode_id for sequence in sequences], dtype=np.int64)
        agents: dict[str, AgentSequenceBatch] = {}
        for agent_id in agent_ids:
            observations = np.stack(
                [[step.observations[agent_id] for step in sequence] for sequence in sequences],
                axis=0,
            )
            actions = np.asarray(
                [[step.actions[agent_id] for step in sequence] for sequence in sequences],
                dtype=np.int64,
            )
            opponent_actions = np.stack(
                [[step.opponent_actions[agent_id] for step in sequence] for sequence in sequences],
                axis=0,
            )
            rewards = np.asarray(
                [[step.rewards[agent_id] for step in sequence] for sequence in sequences],
                dtype=np.float32,
            )
            raw_rewards = np.asarray(
                [[step.raw_rewards[agent_id] for step in sequence] for sequence in sequences],
                dtype=np.float32,
            )
            next_observations = np.stack(
                [[step.next_observations[agent_id] for step in sequence] for sequence in sequences],
                axis=0,
            )
            terminated = np.asarray(
                [[step.terminated[agent_id] for step in sequence] for sequence in sequences],
                dtype=np.float32,
            )
            truncated = np.asarray(
                [[step.truncated[agent_id] for step in sequence] for sequence in sequences],
                dtype=np.float32,
            )
            alive = np.asarray(
                [[step.alive[agent_id] for step in sequence] for sequence in sequences],
                dtype=np.float32,
            )
            agents[agent_id] = AgentSequenceBatch(
                observations=observations,
                actions=actions,
                opponent_actions=opponent_actions,
                rewards=rewards,
                raw_rewards=raw_rewards,
                next_observations=next_observations,
                terminated=terminated,
                truncated=truncated,
                continues=1.0 - np.clip(terminated + truncated, 0.0, 1.0),
                alive=alive,
            )
        return SequenceBatch(episode_ids=episode_ids, agents=agents)

    def __len__(self) -> int:
        return len(self._storage)

    def num_valid_sequences(self, sequence_length: int) -> int:
        return len(self._valid_sequence_starts(sequence_length))

    def _valid_sequence_starts(self, sequence_length: int) -> list[int]:
        storage = list(self._storage)
        valid_starts: list[int] = []
        upper = len(storage) - sequence_length + 1
        for start in range(max(0, upper)):
            episode_id = storage[start].episode_id
            if all(storage[start + offset].episode_id == episode_id for offset in range(sequence_length)):
                valid_starts.append(start)
        return valid_starts


def build_opponent_context(
    agent_id: str,
    agent_ids: tuple[str, ...],
    actions: dict[str, int],
    alive: dict[str, bool],
    action_dim: int,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for other_id in agent_ids:
        if other_id == agent_id:
            continue
        one_hot = np.zeros(action_dim, dtype=np.float32)
        one_hot[int(actions[other_id])] = 1.0
        parts.append(one_hot)
        parts.append(np.asarray([float(alive[other_id])], dtype=np.float32))
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts, axis=0)
