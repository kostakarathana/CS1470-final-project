from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Protocol
import types

import numpy as np

from madreamer.envs.base import ActionDict, EventDict, InfoDict, ObservationDict, StepResult

RIGID_TILE = 1
WOOD_TILE = 2
FLAME_TILE = 4
POWERUP_TILES = {6, 7, 8}
AGENT_BOARD_BASE = 10
POMMERMAN_ACTION_DIM = 6
STOP_ACTION = 0
BOMB_ACTION = 5
STEP_PENALTY = 0.001
SAFE_STOP_PENALTY = 0.01
TIE_PENALTY = 0.25
USEFUL_BOMB_REWARD = 0.03


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
        np.full((1, board_size, board_size), float(observation.get("can_kick", 0)), dtype=np.float32),
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


def extract_pommerman_events(
    previous_observations: dict[str, Mapping[str, Any]],
    next_observations: dict[str, Mapping[str, Any]],
    raw_rewards: dict[str, float],
    terminated: dict[str, bool],
    truncated: dict[str, bool],
    *,
    agent_ids: tuple[str, ...],
    board_size: int,
    mode: str,
    actions: Mapping[str, int] | None = None,
) -> tuple[EventDict, dict[str, bool]]:
    board_before = _coerce_board(previous_observations[agent_ids[0]]["board"], board_size)
    board_after = _coerce_board(next_observations[agent_ids[0]]["board"], board_size)
    alive_before = _alive_from_board(board_before, agent_ids)
    alive_after = _alive_from_board(board_after, agent_ids)
    destroyed_wood = float(np.logical_and(board_before == WOOD_TILE, board_after != WOOD_TILE).sum())
    any_positive_terminal_reward = any(reward > 0.0 for reward in raw_rewards.values())
    tied = all(truncated.values()) or (all(terminated.values()) and not any_positive_terminal_reward)
    events: EventDict = {}
    for agent_id in agent_ids:
        next_position = tuple(int(value) for value in next_observations[agent_id].get("position", (0, 0)))
        powerup_pickups = float(board_before[next_position] in POWERUP_TILES)
        enemy_eliminations = float(
            sum(
                1
                for other_id in agent_ids
                if other_id != agent_id
                and _is_enemy(agent_id, other_id, mode)
                and alive_before[other_id]
                and not alive_after[other_id]
            )
        )
        events[agent_id] = {
            "wood_destroyed": destroyed_wood,
            "powerup_pickups": powerup_pickups,
            "enemy_eliminations": enemy_eliminations,
            "won": float((terminated[agent_id] or truncated[agent_id]) and raw_rewards[agent_id] > 0.0),
            "lost": float((terminated[agent_id] or truncated[agent_id]) and raw_rewards[agent_id] < 0.0),
            "tied": float((terminated[agent_id] or truncated[agent_id]) and tied),
            "safe_stop": 0.0,
            "useful_bomb": 0.0,
        }
        if actions is not None:
            action = int(actions[agent_id])
            previous_observation = previous_observations[agent_id]
            events[agent_id]["safe_stop"] = float(
                action == STOP_ACTION
                and not _is_immediate_bomb_threat(previous_observation, board_size)
            )
            events[agent_id]["useful_bomb"] = float(
                action == BOMB_ACTION
                and _is_useful_bomb_action(
                    previous_observation,
                    agent_id,
                    agent_ids,
                    board_size,
                    mode,
                )
            )
    return events, alive_after


def shape_pommerman_rewards(
    raw_rewards: dict[str, float],
    events: EventDict,
    terminated: dict[str, bool],
    truncated: dict[str, bool],
    *,
    reward_preset: str,
) -> dict[str, float]:
    if reward_preset == "sparse":
        return {
            agent_id: raw_rewards[agent_id] if terminated[agent_id] or truncated[agent_id] else 0.0
            for agent_id in raw_rewards
        }
    if reward_preset != "shaped":
        raise ValueError(f"Unsupported reward preset '{reward_preset}'.")
    shaped: dict[str, float] = {}
    for agent_id, reward in raw_rewards.items():
        event = events[agent_id]
        shaped[agent_id] = (
            1.0 * event["won"]
            - 1.0 * event["lost"]
            + 0.02 * event["wood_destroyed"]
            + 0.05 * event["powerup_pickups"]
            + 0.2 * event["enemy_eliminations"]
            - TIE_PENALTY * event["tied"]
            - STEP_PENALTY
            - SAFE_STOP_PENALTY * event.get("safe_stop", 0.0)
            + USEFUL_BOMB_REWARD * event.get("useful_bomb", 0.0)
        )
        if not (terminated[agent_id] or truncated[agent_id]):
            shaped[agent_id] += 0.0 * reward
    return shaped


@dataclass
class PommermanEnv:
    mode: str = "ffa"
    num_agents: int = 4
    env_id: str = "PommeFFACompetition-v0"
    board_size: int = 11
    max_steps: int = 800
    observability: str = "full"
    communication: bool = False
    board_value_count: int = 14
    reward_preset: str = "sparse"
    backend_factory: Callable[["PommermanEnv"], PommermanBackend] | None = None
    last_infos: InfoDict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("PommermanEnv max_steps must be positive.")
        self.agent_ids = tuple(f"agent_{index}" for index in range(self.num_agents))
        extra_channels = 2 if self.communication else 0
        self.observation_shape = (self.board_value_count + 6 + extra_channels, self.board_size, self.board_size)
        self.action_dim = POMMERMAN_ACTION_DIM
        self._backend = self._make_backend()
        self._last_raw_observations: dict[str, Mapping[str, Any]] = {}
        self._steps = 0

    def reset(self, seed: int | None = None) -> ObservationDict:
        self._steps = 0
        result = _call_reset(self._backend, seed)
        if isinstance(result, tuple) and len(result) == 2 and _looks_like_observation_batch(result[0]):
            raw_observations, infos = result
        else:
            raw_observations, infos = result, {}
        normalized_raw_observations = _normalize_observation_batch(raw_observations, self.agent_ids)
        observations = {
            agent_id: encode_pommerman_observation(
                normalized_raw_observations[agent_id],
                board_size=self.board_size,
                board_value_count=self.board_value_count,
                communication=self.communication,
            )
            for agent_id in self.agent_ids
        }
        self._last_raw_observations = normalized_raw_observations
        self.last_infos = _normalize_info_batch(infos, self.agent_ids)
        for agent_id in self.agent_ids:
            self.last_infos.setdefault(agent_id, {})
            self.last_infos[agent_id]["raw_observation"] = normalized_raw_observations[agent_id]
        return observations

    def step(self, actions: ActionDict) -> StepResult:
        previous_observations = self._last_raw_observations
        backend_actions = [self._normalize_action(actions[agent_id]) for agent_id in self.agent_ids]
        result = self._backend.step(backend_actions)
        self._steps += 1
        raw_observations, raw_rewards, terminated, truncated, infos = _normalize_step_output(result, self.agent_ids)
        if self._steps >= self.max_steps:
            truncated = {
                agent_id: bool(truncated[agent_id] or not terminated[agent_id])
                for agent_id in self.agent_ids
            }
        events, alive = extract_pommerman_events(
            previous_observations,
            raw_observations,
            raw_rewards,
            terminated,
            truncated,
            agent_ids=self.agent_ids,
            board_size=self.board_size,
            mode=self.mode,
            actions=actions,
        )
        rewards = shape_pommerman_rewards(
            raw_rewards,
            events,
            terminated,
            truncated,
            reward_preset=self.reward_preset,
        )
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
            info_by_agent[agent_id]["alive"] = alive[agent_id]
        self.last_infos = info_by_agent
        self._last_raw_observations = raw_observations
        return StepResult(
            observations=observations,
            rewards=rewards,
            raw_rewards=raw_rewards,
            terminated=terminated,
            truncated=truncated,
            alive=alive,
            infos=info_by_agent,
            events=events,
        )

    def close(self) -> None:
        close = getattr(self._backend, "close", None)
        if callable(close):
            close()

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

    def raw_observation(self, agent_id: str) -> Mapping[str, Any]:
        return self._last_raw_observations[agent_id]


def _default_pommerman_backend(env_id: str, num_agents: int) -> PommermanBackend:
    _install_rapidjson_stub()
    _install_pommerman_headless_stubs()
    _add_pommerman_source_paths()
    try:
        pommerman = importlib.import_module("pommerman")
        agents = importlib.import_module("pommerman.agents")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Pommerman is not installed. Install the official playground package, set "
            "POMMERMAN_SOURCE_DIR to a local checkout, or keep the bundled third_party/pommerman source."
        ) from exc
    make = getattr(pommerman, "make", None)
    if make is None:
        raise RuntimeError("Installed pommerman package does not expose pommerman.make.")
    agent_ctor = getattr(agents, "SimpleAgent", None) or getattr(agents, "RandomAgent", None)
    if agent_ctor is None:
        raise RuntimeError("Installed pommerman package does not expose SimpleAgent or RandomAgent constructors.")
    return make(env_id, [agent_ctor() for _ in range(num_agents)])


def _add_pommerman_source_paths() -> None:
    search_roots: list[Path] = []
    source_dir = os.getenv("POMMERMAN_SOURCE_DIR")
    if source_dir:
        search_roots.append(Path(source_dir))
    bundled_root = Path(__file__).resolve().parents[3] / "third_party"
    if (bundled_root / "pommerman").exists():
        search_roots.append(bundled_root)
    for source_path in search_roots:
        if source_path.exists() and str(source_path) not in sys.path:
            sys.path.insert(0, str(source_path))


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
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.window = types.SimpleNamespace(push_handlers=lambda *a, **k: None)

            def set_board(self, *args: object, **kwargs: object) -> None:
                return None

            def set_agents(self, *args: object, **kwargs: object) -> None:
                return None

            def set_step(self, *args: object, **kwargs: object) -> None:
                return None

            def set_bombs(self, *args: object, **kwargs: object) -> None:
                return None

            def render(self, *args: object, **kwargs: object) -> None:
                return None

            def close(self) -> None:
                return None

        class _PixelViewer(_NoopViewer):
            @staticmethod
            def rgb_array(*args: object, **kwargs: object) -> np.ndarray:
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


def _is_useful_bomb_action(
    observation: Mapping[str, Any],
    agent_id: str,
    agent_ids: tuple[str, ...],
    board_size: int,
    mode: str,
) -> bool:
    if int(observation.get("ammo", 0)) <= 0:
        return False
    board = _coerce_board(observation["board"], board_size)
    position = tuple(int(value) for value in observation["position"])
    for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        target = (position[0] + delta_row, position[1] + delta_col)
        if not _in_bounds(board, target):
            continue
        tile = int(board[target])
        if tile == WOOD_TILE:
            return True
        for other_id in agent_ids:
            if other_id != agent_id and _is_enemy(agent_id, other_id, mode) and tile == _agent_board_value(other_id):
                return True
    return False


def _is_immediate_bomb_threat(observation: Mapping[str, Any], board_size: int) -> bool:
    board = _coerce_board(observation["board"], board_size)
    position = tuple(int(value) for value in observation["position"])
    if int(board[position]) == FLAME_TILE:
        return True
    bomb_life = _coerce_board(
        observation.get("bomb_life", np.zeros_like(board)),
        board_size,
    )
    bomb_blast_strength = _coerce_board(
        observation.get("bomb_blast_strength", np.zeros_like(board)),
        board_size,
    )
    for bomb_row, bomb_col in np.argwhere(bomb_life > 0):
        bomb_position = (int(bomb_row), int(bomb_col))
        if bomb_position == position:
            return True
        blast_strength = max(1, int(bomb_blast_strength[bomb_position]))
        if _bomb_hits_position(board, bomb_position, position, blast_strength):
            return True
    return False


def _bomb_hits_position(
    board: np.ndarray,
    bomb_position: tuple[int, int],
    position: tuple[int, int],
    blast_strength: int,
) -> bool:
    if bomb_position[0] != position[0] and bomb_position[1] != position[1]:
        return False
    distance = abs(bomb_position[0] - position[0]) + abs(bomb_position[1] - position[1])
    if distance > blast_strength:
        return False
    row_step = _sign(position[0] - bomb_position[0])
    col_step = _sign(position[1] - bomb_position[1])
    current = (bomb_position[0] + row_step, bomb_position[1] + col_step)
    while current != position:
        if int(board[current]) in {RIGID_TILE, WOOD_TILE}:
            return False
        current = (current[0] + row_step, current[1] + col_step)
    return True


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _in_bounds(board: np.ndarray, position: tuple[int, int]) -> bool:
    return 0 <= position[0] < board.shape[0] and 0 <= position[1] < board.shape[1]


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


def _alive_from_board(board: np.ndarray, agent_ids: tuple[str, ...]) -> dict[str, bool]:
    return {
        agent_id: bool(np.any(board == _agent_board_value(agent_id)))
        for agent_id in agent_ids
    }


def _agent_board_value(agent_id: str) -> int:
    return AGENT_BOARD_BASE + int(agent_id.split("_")[-1])


def _is_enemy(agent_id: str, other_id: str, mode: str) -> bool:
    if mode == "ffa":
        return agent_id != other_id
    team_map = {"agent_0": "agent_3", "agent_3": "agent_0", "agent_1": "agent_2", "agent_2": "agent_1"}
    return other_id not in {agent_id, team_map.get(agent_id)}
