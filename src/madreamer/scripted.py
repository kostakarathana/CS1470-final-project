from __future__ import annotations

import time
from typing import Callable

import numpy as np

from madreamer.envs.mock_grid import MockGridEnv

ScriptedPolicy = Callable[[str, np.ndarray, dict[str, object]], int]


def build_scripted_policy(env: object, seed: int, action_dim: int) -> ScriptedPolicy:
    rng = np.random.default_rng(seed)
    if isinstance(env, MockGridEnv):
        return _mock_grid_greedy_policy
    return lambda _agent_id, _obs, _info: int(rng.integers(0, action_dim))


def _mock_grid_greedy_policy(_agent_id: str, obs: np.ndarray, _info: dict[str, object]) -> int:
    self_pos = np.argwhere(obs[0] > 0.5)[0]
    target_pos = np.argwhere(obs[2] > 0.5)[0]
    row_delta = int(target_pos[0] - self_pos[0])
    col_delta = int(target_pos[1] - self_pos[1])
    if row_delta < 0:
        return 1
    if row_delta > 0:
        return 2
    if col_delta < 0:
        return 3
    if col_delta > 0:
        return 4
    return 0


def render_mock_grid_observation(observation: np.ndarray) -> str:
    height, width = observation.shape[1], observation.shape[2]
    grid = [["." for _ in range(width)] for _ in range(height)]
    self_positions = np.argwhere(observation[0] > 0.5)
    other_positions = np.argwhere(observation[1] > 0.5)
    target_positions = np.argwhere(observation[2] > 0.5)
    for row, col in other_positions:
        grid[int(row)][int(col)] = "o"
    for row, col in target_positions:
        grid[int(row)][int(col)] = "X"
    for row, col in self_positions:
        grid[int(row)][int(col)] = "A"
    return "\n".join(" ".join(row) for row in grid)


def sleep_for_render(delay_seconds: float) -> None:
    if delay_seconds > 0:
        time.sleep(delay_seconds)
