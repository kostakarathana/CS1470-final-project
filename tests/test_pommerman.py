from __future__ import annotations

from pathlib import Path

import numpy as np

from madreamer.config import load_experiment_config
from madreamer.envs.factory import build_env
from madreamer.envs.pommerman import PommermanEnv, encode_pommerman_observation
from madreamer.replay import MultiAgentReplayBuffer
from madreamer.rollout import collect_episode


def _make_observation(agent_index: int, step_count: int = 0) -> dict[str, object]:
    board = np.zeros((11, 11), dtype=np.int64)
    board[agent_index, agent_index] = 10 + agent_index
    board[5, 5] = 2
    bomb_blast_strength = np.zeros((11, 11), dtype=np.float32)
    bomb_life = np.zeros((11, 11), dtype=np.float32)
    if step_count:
        bomb_blast_strength[4, 4] = 3
        bomb_life[4, 4] = 5
    return {
        "board": board,
        "position": (agent_index, agent_index),
        "ammo": 1 + agent_index,
        "blast_strength": 2 + agent_index,
        "can_kick": agent_index % 2,
        "teammate": -1,
        "enemies": [value for value in range(4) if value != agent_index],
        "bomb_blast_strength": bomb_blast_strength,
        "bomb_life": bomb_life,
        "message": (0, 0),
    }


class FakePommermanBackend:
    def __init__(self) -> None:
        self.step_count = 0
        self.last_actions: list[int] = []

    def reset(self, seed: int | None = None) -> list[dict[str, object]]:
        self.step_count = 0
        return [_make_observation(agent_index) for agent_index in range(4)]

    def step(self, actions: list[int]) -> tuple[list[dict[str, object]], list[float], bool, dict[str, object]]:
        self.step_count += 1
        self.last_actions = actions
        observations = [_make_observation(agent_index, step_count=self.step_count) for agent_index in range(4)]
        rewards = [1.0 if index == 0 and self.step_count == 2 else 0.0 for index in range(4)]
        done = self.step_count >= 2
        info = {"step_count": self.step_count}
        return observations, rewards, done, info

    def close(self) -> None:
        return None


def _backend_factory(_: PommermanEnv) -> FakePommermanBackend:
    return FakePommermanBackend()


def test_encode_pommerman_observation_shapes_and_scalars() -> None:
    encoded = encode_pommerman_observation(
        _make_observation(1, step_count=1),
        board_size=11,
        board_value_count=14,
        communication=False,
    )

    assert encoded.shape == (20, 11, 11)
    assert encoded[11, 1, 1] == 1.0
    assert encoded[14, 4, 4] == 3.0 / 11.0
    assert encoded[15, 4, 4] == 0.5
    assert encoded[16, 1, 1] == 1.0
    assert np.allclose(encoded[17], 0.2)
    assert np.allclose(encoded[18], 0.3)


def test_pommerman_env_adapter_reset_and_step() -> None:
    env = PommermanEnv(
        num_agents=4,
        board_size=11,
        communication=False,
        backend_factory=_backend_factory,
    )

    observations = env.reset(seed=123)
    assert set(observations) == {"agent_0", "agent_1", "agent_2", "agent_3"}
    assert observations["agent_0"].shape == (20, 11, 11)

    step = env.step({agent_id: 0 for agent_id in env.agent_ids})
    assert step.rewards["agent_0"] == 0.0
    assert step.terminated["agent_0"] is False
    assert "raw_observation" in step.infos["agent_0"]


def test_collect_episode_writes_replay_transitions() -> None:
    env = PommermanEnv(
        num_agents=4,
        board_size=11,
        communication=False,
        backend_factory=_backend_factory,
    )
    replay = MultiAgentReplayBuffer(capacity=8)

    policies = {agent_id: (lambda _agent_id, _obs, _info: 0) for agent_id in env.agent_ids}
    summary = collect_episode(env, policies, replay=replay, seed=0)

    assert summary.steps == 2
    assert summary.done is True
    assert len(replay) == 2
    transition = replay.sample(1)[0]
    assert transition.actions["agent_0"] == 0
    assert transition.infos is not None
    assert transition.infos["agent_0"]["shared_info"]["step_count"] in {1, 2}


def test_build_env_selects_pommerman_adapter() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "pommerman_phase1.yaml"
    cfg = load_experiment_config(config_path)
    env = build_env(cfg, pommerman_backend_factory=_backend_factory)

    assert isinstance(env, PommermanEnv)
    assert env.observation_shape == (20, 11, 11)


def test_real_pommerman_env_reset_and_step() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "pommerman_phase1.yaml"
    cfg = load_experiment_config(config_path)
    env = build_env(cfg)

    observations = env.reset(seed=0)
    assert observations["agent_0"].shape == (20, 11, 11)

    step = env.step({agent_id: 0 for agent_id in env.agent_ids})
    assert set(step.rewards) == set(env.agent_ids)
    assert "raw_observation" in step.infos["agent_0"]
