from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from madreamer.envs.base import ActionDict, InfoDict, ObservationDict, StepResult


@dataclass
class MockGridEnv:
    num_agents: int = 4
    grid_size: int = 5
    max_steps: int = 16
    task_type: str = "cooperative"
    reward_preset: str = "sparse"

    def __post_init__(self) -> None:
        if self.num_agents < 2:
            raise ValueError("MockGridEnv expects at least two agents.")
        self.agent_ids = tuple(f"agent_{index}" for index in range(self.num_agents))
        self.observation_shape = (3, self.grid_size, self.grid_size)
        self.action_dim = 5
        self._rng = np.random.default_rng()
        self._target = (self.grid_size // 2, self.grid_size // 2)
        self._steps = 0
        self._positions: dict[str, tuple[int, int]] = {}
        self.last_infos: InfoDict = {}

    def reset(self, seed: int | None = None) -> ObservationDict:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._steps = 0
        taken: set[tuple[int, int]] = {self._target}
        self._positions = {}
        for agent_id in self.agent_ids:
            while True:
                position = (
                    int(self._rng.integers(0, self.grid_size)),
                    int(self._rng.integers(0, self.grid_size)),
                )
                if position not in taken:
                    taken.add(position)
                    self._positions[agent_id] = position
                    break
        observations = {agent_id: self._make_observation(agent_id) for agent_id in self.agent_ids}
        self.last_infos = {
            agent_id: {
                "position": self._positions[agent_id],
                "target": self._target,
                "step": self._steps,
                "raw_observation": observations[agent_id].copy(),
            }
            for agent_id in self.agent_ids
        }
        return observations

    def step(self, actions: ActionDict) -> StepResult:
        self._steps += 1
        for agent_id, action in actions.items():
            self._positions[agent_id] = self._apply_action(self._positions[agent_id], action)

        raw_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
        terminated = {agent_id: False for agent_id in self.agent_ids}
        truncated = {agent_id: self._steps >= self.max_steps for agent_id in self.agent_ids}
        alive = {agent_id: True for agent_id in self.agent_ids}
        events = {
            agent_id: {
                "wood_destroyed": 0.0,
                "powerup_pickups": 0.0,
                "enemy_eliminations": 0.0,
                "won": 0.0,
                "lost": 0.0,
                "tied": 0.0,
            }
            for agent_id in self.agent_ids
        }

        winners = [agent_id for agent_id, position in self._positions.items() if position == self._target]
        if winners:
            if self.task_type == "cooperative":
                raw_rewards = {agent_id: 1.0 for agent_id in self.agent_ids}
            else:
                raw_rewards = {
                    agent_id: (1.0 if agent_id in winners else -1.0) for agent_id in self.agent_ids
                }
            terminated = {agent_id: True for agent_id in self.agent_ids}
            for agent_id in self.agent_ids:
                events[agent_id]["won"] = float(raw_rewards[agent_id] > 0.0)
                events[agent_id]["lost"] = float(raw_rewards[agent_id] < 0.0)
        elif all(truncated.values()):
            for agent_id in self.agent_ids:
                events[agent_id]["tied"] = 1.0

        observations = {agent_id: self._make_observation(agent_id) for agent_id in self.agent_ids}
        infos: InfoDict = {
            agent_id: {
                "position": self._positions[agent_id],
                "target": self._target,
                "step": self._steps,
                "raw_observation": observations[agent_id].copy(),
            }
            for agent_id in self.agent_ids
        }
        rewards = self._shape_rewards(raw_rewards, events, terminated, truncated)
        self.last_infos = infos
        return StepResult(
            observations=observations,
            rewards=rewards,
            raw_rewards=raw_rewards,
            terminated=terminated,
            truncated=truncated,
            alive=alive,
            infos=infos,
            events=events,
        )

    def _apply_action(self, position: tuple[int, int], action: int) -> tuple[int, int]:
        row, col = position
        if action == 1:
            row -= 1
        elif action == 2:
            row += 1
        elif action == 3:
            col -= 1
        elif action == 4:
            col += 1
        row = int(np.clip(row, 0, self.grid_size - 1))
        col = int(np.clip(col, 0, self.grid_size - 1))
        return row, col

    def _make_observation(self, agent_id: str) -> np.ndarray:
        obs = np.zeros(self.observation_shape, dtype=np.float32)
        self_row, self_col = self._positions[agent_id]
        obs[0, self_row, self_col] = 1.0
        for other_id, (other_row, other_col) in self._positions.items():
            if other_id != agent_id:
                obs[1, other_row, other_col] = 1.0
        target_row, target_col = self._target
        obs[2, target_row, target_col] = 1.0
        return obs

    def close(self) -> None:
        return None

    def _shape_rewards(
        self,
        raw_rewards: dict[str, float],
        events: dict[str, dict[str, float]],
        terminated: dict[str, bool],
        truncated: dict[str, bool],
    ) -> dict[str, float]:
        if self.reward_preset == "sparse":
            return {
                agent_id: raw_rewards[agent_id] if terminated[agent_id] or truncated[agent_id] else 0.0
                for agent_id in self.agent_ids
            }
        shaped = {agent_id: 0.0 for agent_id in self.agent_ids}
        for agent_id in self.agent_ids:
            shaped[agent_id] += events[agent_id]["won"]
            shaped[agent_id] -= events[agent_id]["lost"]
        return shaped
