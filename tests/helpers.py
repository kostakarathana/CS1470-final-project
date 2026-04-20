from __future__ import annotations

from collections.abc import Callable

import numpy as np

from madreamer.envs.pommerman import PommermanEnv


def make_observation(
    agent_index: int,
    *,
    board: np.ndarray | None = None,
    position: tuple[int, int] | None = None,
    ammo: int = 1,
    blast_strength: int = 2,
    can_kick: int = 0,
) -> dict[str, object]:
    board_size = 11
    obs_board = np.zeros((board_size, board_size), dtype=np.int64) if board is None else np.asarray(board).copy()
    pos = position or (agent_index, agent_index)
    obs_board[pos] = 10 + agent_index
    bomb_blast_strength = np.zeros((board_size, board_size), dtype=np.float32)
    bomb_life = np.zeros((board_size, board_size), dtype=np.float32)
    return {
        "board": obs_board,
        "position": pos,
        "ammo": ammo,
        "blast_strength": blast_strength,
        "can_kick": can_kick,
        "teammate": -1,
        "enemies": [10 + idx for idx in range(4) if idx != agent_index],
        "bomb_blast_strength": bomb_blast_strength,
        "bomb_life": bomb_life,
        "message": (0, 0),
    }


class FakePommermanBackend:
    def __init__(self, mode: str = "ffa") -> None:
        self.mode = mode
        self.step_count = 0
        self.last_actions: list[int] = []

    def reset(self, seed: int | None = None) -> list[dict[str, object]]:
        del seed
        self.step_count = 0
        board = np.zeros((11, 11), dtype=np.int64)
        board[5, 5] = 2
        return [
            make_observation(0, board=board, position=(0, 0), ammo=1),
            make_observation(1, board=board, position=(0, 10), ammo=1),
            make_observation(2, board=board, position=(10, 0), ammo=1),
            make_observation(3, board=board, position=(10, 10), ammo=1),
        ]

    def step(self, actions: list[int]) -> tuple[list[dict[str, object]], list[float], bool, dict[str, object]]:
        self.step_count += 1
        self.last_actions = actions
        board = np.zeros((11, 11), dtype=np.int64)
        board[4, 4] = 6
        if self.step_count < 3:
            board[5, 5] = 0
            observations = [
                make_observation(0, board=board, position=(4, 4), ammo=1, can_kick=1),
                make_observation(1, board=board, position=(0, 10), ammo=1),
                make_observation(2, board=board, position=(10, 0), ammo=1),
                make_observation(3, board=board, position=(10, 10), ammo=1),
            ]
            rewards = [0.0, 0.0, 0.0, 0.0]
            done = False
        else:
            if self.mode == "team":
                observations = [
                    make_observation(0, board=board, position=(4, 4), ammo=1, can_kick=1),
                    make_observation(1, board=board, position=(0, 10), ammo=1),
                    make_observation(2, board=board, position=(10, 0), ammo=1),
                    make_observation(3, board=board, position=(10, 10), ammo=1),
                ]
                rewards = [1.0, -1.0, -1.0, 1.0]
            else:
                observations = [
                    make_observation(0, board=board, position=(4, 4), ammo=1, can_kick=1),
                    make_observation(1, board=board * 0, position=(0, 10), ammo=1),
                    make_observation(2, board=board * 0, position=(10, 0), ammo=1),
                    make_observation(3, board=board * 0, position=(10, 10), ammo=1),
                ]
                observations[1]["board"] = np.zeros((11, 11), dtype=np.int64)
                observations[2]["board"] = np.zeros((11, 11), dtype=np.int64)
                observations[3]["board"] = np.zeros((11, 11), dtype=np.int64)
                rewards = [1.0, -1.0, -1.0, -1.0]
            done = True
        return observations, rewards, done, {"step_count": self.step_count}

    def close(self) -> None:
        return None


def fake_backend_factory(mode: str = "ffa") -> Callable[[PommermanEnv], FakePommermanBackend]:
    def _factory(env: PommermanEnv) -> FakePommermanBackend:
        return FakePommermanBackend(mode=mode or env.mode)

    return _factory
