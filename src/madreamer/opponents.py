from __future__ import annotations

from collections.abc import Mapping
import importlib
from typing import Any

import numpy as np


PASSABLE_TILES = {0, 5, 6, 7, 8}
WOOD_TILE = 2
BOMB_ACTION = 5
STOP_ACTION = 0
MOVE_ACTIONS = {
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}


class FixedOpponentManager:
    def __init__(
        self,
        env: Any,
        *,
        policy_name: str,
        controlled_agent_ids: tuple[str, ...],
        seed: int,
    ) -> None:
        self.env = env
        self.policy_name = policy_name
        self.controlled_agent_ids = set(controlled_agent_ids)
        self.rng = np.random.default_rng(seed)
        self._official_agents = self._build_official_agents(policy_name)

    def actions(
        self,
        observations: dict[str, np.ndarray],
        infos: dict[str, dict[str, object]],
    ) -> dict[str, int]:
        actions: dict[str, int] = {}
        for agent_id in self.env.agent_ids:
            if agent_id in self.controlled_agent_ids:
                continue
            info = infos.get(agent_id, {})
            raw_observation = info.get("raw_observation")
            if self.policy_name == "noop":
                actions[agent_id] = STOP_ACTION
            elif self.policy_name == "random":
                actions[agent_id] = int(self.rng.integers(0, self.env.action_dim))
            elif isinstance(raw_observation, Mapping):
                actions[agent_id] = self._pommerman_action(agent_id, raw_observation)
            else:
                actions[agent_id] = self._mock_grid_action(observations[agent_id])
        return actions

    def _build_official_agents(self, policy_name: str) -> dict[str, object]:
        if policy_name not in {"simple", "random"}:
            return {}
        try:
            agents = importlib.import_module("pommerman.agents")
        except ModuleNotFoundError:
            return {}
        ctor_name = "SimpleAgent" if policy_name == "simple" else "RandomAgent"
        ctor = getattr(agents, ctor_name, None)
        if ctor is None:
            return {}
        return {
            agent_id: ctor()
            for agent_id in getattr(self.env, "agent_ids", ())
            if agent_id not in self.controlled_agent_ids
        }

    def _pommerman_action(self, agent_id: str, raw_observation: Mapping[str, object]) -> int:
        official_agent = self._official_agents.get(agent_id)
        if official_agent is not None:
            try:
                return int(official_agent.act(raw_observation, getattr(self.env, "action_dim", 6)))
            except Exception:
                pass
        return _simple_pommerman_heuristic(raw_observation, self.rng)

    def _mock_grid_action(self, observation: np.ndarray) -> int:
        self_pos = tuple(int(value) for value in np.argwhere(observation[0] > 0.5)[0])
        target_pos = tuple(int(value) for value in np.argwhere(observation[2] > 0.5)[0])
        if target_pos[0] < self_pos[0]:
            return 1
        if target_pos[0] > self_pos[0]:
            return 2
        if target_pos[1] < self_pos[1]:
            return 3
        if target_pos[1] > self_pos[1]:
            return 4
        return STOP_ACTION


def _simple_pommerman_heuristic(raw_observation: Mapping[str, object], rng: np.random.Generator) -> int:
    board = np.asarray(raw_observation["board"])
    bomb_life = np.asarray(raw_observation.get("bomb_life", np.zeros_like(board)))
    position = tuple(int(value) for value in raw_observation["position"])
    ammo = int(raw_observation.get("ammo", 0))
    if bomb_life[position] > 0:
        safe_actions = _safe_moves(board, bomb_life, position)
        if safe_actions:
            return int(rng.choice(safe_actions))
    if ammo > 0 and _has_adjacent_target(board, position):
        return BOMB_ACTION
    passable_actions = _passable_moves(board, position)
    if passable_actions:
        return int(rng.choice(passable_actions))
    return STOP_ACTION


def _safe_moves(board: np.ndarray, bomb_life: np.ndarray, position: tuple[int, int]) -> list[int]:
    safe: list[int] = []
    for action, delta in MOVE_ACTIONS.items():
        next_pos = (position[0] + delta[0], position[1] + delta[1])
        if _is_passable(board, next_pos) and bomb_life[next_pos] <= 0:
            safe.append(action)
    return safe


def _passable_moves(board: np.ndarray, position: tuple[int, int]) -> list[int]:
    actions: list[int] = []
    for action, delta in MOVE_ACTIONS.items():
        next_pos = (position[0] + delta[0], position[1] + delta[1])
        if _is_passable(board, next_pos):
            actions.append(action)
    return actions


def _has_adjacent_target(board: np.ndarray, position: tuple[int, int]) -> bool:
    for delta in MOVE_ACTIONS.values():
        next_pos = (position[0] + delta[0], position[1] + delta[1])
        if not _in_bounds(board, next_pos):
            continue
        tile = int(board[next_pos])
        if tile == WOOD_TILE or tile >= 10:
            return True
    return False


def _is_passable(board: np.ndarray, position: tuple[int, int]) -> bool:
    return _in_bounds(board, position) and int(board[position]) in PASSABLE_TILES


def _in_bounds(board: np.ndarray, position: tuple[int, int]) -> bool:
    return 0 <= position[0] < board.shape[0] and 0 <= position[1] < board.shape[1]
