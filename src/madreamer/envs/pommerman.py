from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Protocol
import types

import numpy as np

from madreamer.envs.base import ActionDict, InfoDict, ObservationDict, StepResult


class PommermanBackend(Protocol):
    def reset(self, seed: int | None = None) -> Any:
        ...

    def step(self, actions: list[int] | list[tuple[int, int, int]]) -> Any:
        ...

    def close(self) -> None:
        ...


def encode_pommerman_observation(
    observation: Mapping[str, Any],
    *,
    board_size: int,
    board_value_count: int,
    communication: bool,
) -> np.ndarray:
    board = _coerce_board(observation["board"], board_size).astype(np.int64, copy=False)
    board_planes = _one_hot_board(board, board_value_count)

    bomb_blast_strength = _coerce_board(
        observation.get("bomb_blast_strength", np.zeros_like(board)),
        board_size,
    ).astype(np.float32, copy=False)
    bomb_life = _coerce_board(
        observation.get("bomb_life", np.zeros_like(board)),
        board_size,
    ).astype(np.float32, copy=False)

    feature_planes = [
        board_planes,
        bomb_blast_strength[None] / max(1.0, float(board_size)),
        bomb_life[None] / 10.0,
        _position_plane(tuple(observation["position"]), board_size)[None],
        np.full((1, board_size, board_size), float(observation.get("ammo", 0)) / 10.0, dtype=np.float32),
        np.full(
            (1, board_size, board_size),
            float(observation.get("blast_strength", 0)) / 10.0,
            dtype=np.float32,
        ),
        np.full(
            (1, board_size, board_size),
            float(observation.get("can_kick", 0)),
            dtype=np.float32,
        ),
    ]

    if communication:
        message = observation.get("message", (0, 0))
        if len(message) != 2:
            raise ValueError("Pommerman communication expects exactly two message tokens.")
        feature_planes.extend(
            [
                np.full((1, board_size, board_size), float(message[0]) / 8.0, dtype=np.float32),
                np.full((1, board_size, board_size), float(message[1]) / 8.0, dtype=np.float32),
            ]
        )

    return np.concatenate(feature_planes, axis=0).astype(np.float32, copy=False)


@dataclass
class PommermanEnv:
    num_agents: int = 4
    env_id: str = "PommeFFACompetition-v0"
    board_size: int = 11
    max_steps: int = 800
    observability: str = "full"
    communication: bool = False
    board_value_count: int = 14
    opponent_mode: str = "fixed"
    backend_factory: Callable[["PommermanEnv"], PommermanBackend] | None = None

    def __post_init__(self) -> None:
        self.agent_ids = tuple(f"agent_{index}" for index in range(self.num_agents))
        extra_channels = 2 if self.communication else 0
        self.observation_shape = (self.board_value_count + 6 + extra_channels, self.board_size, self.board_size)
        self.action_dim = 6
        self.last_raw_observations: dict[str, Mapping[str, Any]] = {}
        self._backend = self._make_backend()

    def reset(self, seed: int | None = None) -> ObservationDict:
        result = _call_reset(self._backend, seed)
        if isinstance(result, tuple) and len(result) == 2 and _looks_like_observation_batch(result[0]):
            raw_observations, _ = result
        else:
            raw_observations = result
        observations_by_agent = _normalize_observation_batch(raw_observations, self.agent_ids)
        self.last_raw_observations = observations_by_agent
        return {
            agent_id: encode_pommerman_observation(
                observations_by_agent[agent_id],
                board_size=self.board_size,
                board_value_count=self.board_value_count,
                communication=self.communication,
            )
            for agent_id in self.agent_ids
        }

    def step(self, actions: ActionDict) -> StepResult:
        backend_actions = [self._normalize_action(actions[agent_id]) for agent_id in self.agent_ids]
        result = self._backend.step(backend_actions)
        raw_observations, rewards, terminated, truncated, infos = _normalize_step_output(result, self.agent_ids)
        self.last_raw_observations = raw_observations
        observations = {
            agent_id: encode_pommerman_observation(
                raw_observations[agent_id],
                board_size=self.board_size,
                board_value_count=self.board_value_count,
                communication=self.communication,
            )
            for agent_id in self.agent_ids
        }
        info_by_agent = _normalize_info_batch(infos, self.agent_ids)
        for agent_id in self.agent_ids:
            info_by_agent[agent_id]["raw_observation"] = raw_observations[agent_id]
        return StepResult(
            observations=observations,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            infos=info_by_agent,
        )

    def close(self) -> None:
        close = getattr(self._backend, "close", None)
        if callable(close):
            close()

    def render_text(self, agent_id: str = "agent_0") -> str:
        if agent_id not in self.last_raw_observations:
            return "<no Pommerman observation loaded>"
        board = np.asarray(self.last_raw_observations[agent_id]["board"])
        glyphs = {
            0: ".",
            1: "#",
            2: "+",
            3: "B",
            4: "*",
            5: "?",
            6: "b",
            7: "r",
            8: "k",
            9: "d",
            10: "0",
            11: "1",
            12: "2",
            13: "3",
        }
        rows = []
        for row in board:
            rows.append(" ".join(glyphs.get(int(value), str(int(value))) for value in row))
        return "\n".join(rows)

    def _make_backend(self) -> PommermanBackend:
        if self.backend_factory is not None:
            return self.backend_factory(self)
        return _default_pommerman_backend(self.env_id, self.num_agents)

    def _normalize_action(self, action: int | Sequence[int]) -> int | tuple[int, int, int]:
        if isinstance(action, Sequence) and not isinstance(action, (str, bytes)):
            values = tuple(int(value) for value in action)
            if self.communication and len(values) == 3:
                return values
            if len(values) == 1:
                return values[0]
            raise ValueError("Pommerman actions must be an int or a 3-tuple when communication is enabled.")
        return int(action)


def _default_pommerman_backend(env_id: str, num_agents: int) -> PommermanBackend:
    _install_rapidjson_stub()
    _install_pommerman_headless_stubs()
    source_dir = os.getenv("POMMERMAN_SOURCE_DIR")
    search_roots = []
    if source_dir:
        search_roots.append(Path(source_dir))
    bundled_root = Path(__file__).resolve().parents[3] / "third_party"
    if bundled_root.exists():
        search_roots.append(bundled_root)
    for source_path in search_roots:
        if source_path.exists() and str(source_path) not in sys.path:
            sys.path.insert(0, str(source_path))
    try:
        pommerman = importlib.import_module("pommerman")
        agents = importlib.import_module("pommerman.agents")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Pommerman is not installed. Install the package, or set POMMERMAN_SOURCE_DIR to an upstream checkout, or pass a backend_factory to PommermanEnv."
        ) from exc

    make = getattr(pommerman, "make", None)
    if make is None:
        raise RuntimeError("Installed pommerman package does not expose pommerman.make.")

    agent_ctor = getattr(agents, "SimpleAgent", None) or getattr(agents, "RandomAgent", None)
    if agent_ctor is None:
        raise RuntimeError(
            "Installed pommerman package does not expose SimpleAgent or RandomAgent constructors."
        )

    agent_list = [agent_ctor() for _ in range(num_agents)]
    return make(env_id, agent_list)


def _install_rapidjson_stub() -> None:
    if "rapidjson" in sys.modules:
        return
    module = types.ModuleType("rapidjson")
    module.dumps = lambda obj, *args, **kwargs: json.dumps(obj, *args, **kwargs)
    module.loads = lambda value, *args, **kwargs: json.loads(value, *args, **kwargs)
    module.dump = lambda obj, fp, *args, **kwargs: json.dump(obj, fp, *args, **kwargs)
    module.load = lambda fp, *args, **kwargs: json.load(fp, *args, **kwargs)
    sys.modules["rapidjson"] = module


def _install_pommerman_headless_stubs() -> None:
    if "pommerman.graphics" not in sys.modules:
        graphics_module = types.ModuleType("pommerman.graphics")

        class _NoopViewer:
            def __init__(self, *args, **kwargs) -> None:
                self.window = types.SimpleNamespace(push_handlers=lambda *a, **k: None)

            def set_board(self, *args, **kwargs) -> None:
                return None

            def set_agents(self, *args, **kwargs) -> None:
                return None

            def set_step(self, *args, **kwargs) -> None:
                return None

            def set_bombs(self, *args, **kwargs) -> None:
                return None

            def render(self, *args, **kwargs) -> None:
                return None

            def close(self) -> None:
                return None

        class _PixelViewer(_NoopViewer):
            @staticmethod
            def rgb_array(*args, **kwargs):
                return np.zeros((1, 64, 64, 3), dtype=np.uint8)

        graphics_module.PixelViewer = _PixelViewer
        graphics_module.PommeViewer = _NoopViewer
        sys.modules["pommerman.graphics"] = graphics_module

    for module_name in ("pommerman.network", "pommerman.cli"):
        if module_name not in sys.modules:
            sys.modules[module_name] = types.ModuleType(module_name)


def _call_reset(backend: PommermanBackend, seed: int | None) -> Any:
    try:
        return backend.reset(seed=seed)
    except TypeError:
        return backend.reset()


def _coerce_board(value: Any, board_size: int) -> np.ndarray:
    board = np.asarray(value)
    if board.ndim == 1:
        expected = board_size * board_size
        if board.size != expected:
            raise ValueError(f"Expected flattened board of size {expected}, got {board.size}.")
        board = board.reshape(board_size, board_size)
    if board.shape != (board_size, board_size):
        raise ValueError(f"Expected board shape {(board_size, board_size)}, got {board.shape}.")
    return board


def _one_hot_board(board: np.ndarray, board_value_count: int) -> np.ndarray:
    clipped = np.clip(board, 0, board_value_count - 1)
    eye = np.eye(board_value_count, dtype=np.float32)
    return eye[clipped].transpose(2, 0, 1)


def _position_plane(position: tuple[int, int], board_size: int) -> np.ndarray:
    plane = np.zeros((board_size, board_size), dtype=np.float32)
    row, col = position
    plane[int(row), int(col)] = 1.0
    return plane


def _looks_like_observation_batch(value: Any) -> bool:
    return isinstance(value, (Sequence, Mapping)) and not isinstance(value, (str, bytes))


def _normalize_observation_batch(
    observations: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    agent_ids: tuple[str, ...],
) -> dict[str, Mapping[str, Any]]:
    if isinstance(observations, Mapping):
        return {agent_id: observations[agent_id] for agent_id in agent_ids}
    if len(observations) != len(agent_ids):
        raise ValueError("Observation count does not match number of agent ids.")
    return {agent_id: observations[index] for index, agent_id in enumerate(agent_ids)}


def _normalize_step_output(
    result: Any,
    agent_ids: tuple[str, ...],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, float], dict[str, bool], dict[str, bool], Any]:
    if not isinstance(result, tuple):
        raise ValueError("Pommerman backend step output must be a tuple.")
    if len(result) == 4:
        observations, rewards, done, infos = result
        terminated = {agent_id: bool(done) for agent_id in agent_ids}
        truncated = {agent_id: False for agent_id in agent_ids}
    elif len(result) == 5:
        observations, rewards, terminated_raw, truncated_raw, infos = result
        terminated = _coerce_done_flags(terminated_raw, agent_ids)
        truncated = _coerce_done_flags(truncated_raw, agent_ids)
    else:
        raise ValueError("Pommerman backend step output must have 4 or 5 elements.")

    return (
        _normalize_observation_batch(observations, agent_ids),
        _coerce_reward_dict(rewards, agent_ids),
        terminated,
        truncated,
        infos,
    )


def _coerce_reward_dict(
    rewards: Sequence[float] | Mapping[str, float],
    agent_ids: tuple[str, ...],
) -> dict[str, float]:
    if isinstance(rewards, Mapping):
        return {agent_id: float(rewards[agent_id]) for agent_id in agent_ids}
    if len(rewards) != len(agent_ids):
        raise ValueError("Reward count does not match number of agents.")
    return {agent_id: float(rewards[index]) for index, agent_id in enumerate(agent_ids)}


def _coerce_done_flags(
    flags: bool | Sequence[bool] | Mapping[str, bool],
    agent_ids: tuple[str, ...],
) -> dict[str, bool]:
    if isinstance(flags, Mapping):
        return {agent_id: bool(flags[agent_id]) for agent_id in agent_ids}
    if isinstance(flags, Sequence) and not isinstance(flags, (str, bytes)):
        if len(flags) != len(agent_ids):
            raise ValueError("Done flag count does not match number of agents.")
        return {agent_id: bool(flags[index]) for index, agent_id in enumerate(agent_ids)}
    return {agent_id: bool(flags) for agent_id in agent_ids}


def _normalize_info_batch(infos: Any, agent_ids: tuple[str, ...]) -> InfoDict:
    if isinstance(infos, Mapping) and all(agent_id in infos for agent_id in agent_ids):
        normalized: InfoDict = {}
        for agent_id in agent_ids:
            value = infos[agent_id]
            normalized[agent_id] = dict(value) if isinstance(value, Mapping) else {"value": value}
        return normalized
    return {agent_id: {"shared_info": infos} for agent_id in agent_ids}
