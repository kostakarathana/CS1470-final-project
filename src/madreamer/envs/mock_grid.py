from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from madreamer.envs.base import ActionDict, ObservationDict, StepResult


@dataclass
class MockGridEnv:
    num_agents: int = 2
    grid_size: int = 5
    max_steps: int = 16
    task_type: str = "cooperative"

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
        return {agent_id: self._make_observation(agent_id) for agent_id in self.agent_ids}

    def step(self, actions: ActionDict) -> StepResult:
        self._steps += 1
        for agent_id, action in actions.items():
            self._positions[agent_id] = self._apply_action(self._positions[agent_id], action)

        rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
        terminated = {agent_id: False for agent_id in self.agent_ids}
        truncated = {agent_id: self._steps >= self.max_steps for agent_id in self.agent_ids}

        winners = [agent_id for agent_id, position in self._positions.items() if position == self._target]
        if winners:
            if self.task_type == "cooperative":
                rewards = {agent_id: 1.0 for agent_id in self.agent_ids}
            else:
                rewards = {
                    agent_id: (1.0 if agent_id in winners else -1.0) for agent_id in self.agent_ids
                }
            terminated = {agent_id: True for agent_id in self.agent_ids}

        observations = {agent_id: self._make_observation(agent_id) for agent_id in self.agent_ids}
        infos = {
            agent_id: {
                "position": self._positions[agent_id],
                "target": self._target,
                "step": self._steps,
            }
            for agent_id in self.agent_ids
        }
        return StepResult(
            observations=observations,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
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

    def render_text(self, agent_id: str = "agent_0") -> str:
        obs = self._make_observation(agent_id)
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for row, col in np.argwhere(obs[1] > 0.5):
            grid[int(row)][int(col)] = "o"
        for row, col in np.argwhere(obs[2] > 0.5):
            grid[int(row)][int(col)] = "X"
        for row, col in np.argwhere(obs[0] > 0.5):
            grid[int(row)][int(col)] = "A"
        return "\n".join(" ".join(row) for row in grid)
