from __future__ import annotations

from pathlib import Path

import numpy as np

from madreamer.config import load_experiment_config
from madreamer.envs.factory import build_env
from madreamer.envs.pommerman import (
    PommermanEnv,
    encode_pommerman_observation,
    extract_pommerman_events,
    shape_pommerman_rewards,
)

from tests.helpers import fake_backend_factory, make_observation


def test_encode_pommerman_observation_shapes_and_scalars() -> None:
    board = np.zeros((11, 11), dtype=np.int64)
    board[1, 1] = 10
    board[4, 4] = 6
    encoded = encode_pommerman_observation(
        make_observation(0, board=board, position=(1, 1), ammo=2, blast_strength=3, can_kick=1),
        board_size=11,
        board_value_count=14,
        communication=False,
    )

    assert encoded.shape == (20, 11, 11)
    assert encoded[10, 1, 1] == 1.0
    assert encoded[16, 1, 1] == 1.0
    assert np.allclose(encoded[17], 0.2)
    assert np.allclose(encoded[18], 0.3)


def test_extract_events_and_shaped_rewards() -> None:
    board_before = np.zeros((11, 11), dtype=np.int64)
    board_before[0, 0] = 10
    board_before[0, 10] = 11
    board_before[10, 0] = 12
    board_before[10, 10] = 13
    board_before[5, 5] = 2
    board_before[4, 4] = 6
    board_after = board_before.copy()
    board_after[5, 5] = 0
    board_after[4, 4] = 10
    board_after[0, 10] = 0
    previous = {
        f"agent_{idx}": make_observation(idx, board=board_before, position=position)
        for idx, position in enumerate(((0, 0), (0, 10), (10, 0), (10, 10)))
    }
    next_obs = {
        "agent_0": make_observation(0, board=board_after, position=(4, 4), ammo=1, can_kick=1),
        "agent_1": make_observation(1, board=board_after, position=(0, 10)),
        "agent_2": make_observation(2, board=board_after, position=(10, 0)),
        "agent_3": make_observation(3, board=board_after, position=(10, 10)),
    }
    raw_rewards = {"agent_0": 1.0, "agent_1": -1.0, "agent_2": -1.0, "agent_3": -1.0}
    terminated = {agent_id: True for agent_id in raw_rewards}
    truncated = {agent_id: False for agent_id in raw_rewards}
    events, alive = extract_pommerman_events(
        previous,
        next_obs,
        raw_rewards,
        terminated,
        truncated,
        agent_ids=tuple(previous),
        board_size=11,
        mode="ffa",
    )

    assert alive["agent_1"] is False
    assert events["agent_0"]["wood_destroyed"] == 1.0
    assert events["agent_0"]["powerup_pickups"] == 1.0
    assert events["agent_0"]["enemy_eliminations"] >= 1.0
    shaped = shape_pommerman_rewards(raw_rewards, events, terminated, truncated, reward_preset="shaped")
    assert shaped["agent_0"] > 1.0


def test_pommerman_env_adapter_reset_and_step() -> None:
    env = PommermanEnv(
        num_agents=4,
        board_size=11,
        communication=False,
        backend_factory=fake_backend_factory("ffa"),
    )

    observations = env.reset(seed=123)
    assert set(observations) == {"agent_0", "agent_1", "agent_2", "agent_3"}
    assert observations["agent_0"].shape == (20, 11, 11)

    step = env.step({agent_id: 0 for agent_id in env.agent_ids})
    assert step.rewards["agent_0"] >= 0.0
    assert "raw_observation" in step.infos["agent_0"]
    assert "wood_destroyed" in step.events["agent_0"]


def test_build_env_selects_pommerman_adapter() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "shared_smoke.yaml"
    cfg = load_experiment_config(config_path)
    env = build_env(cfg, pommerman_backend_factory=fake_backend_factory("ffa"))

    assert isinstance(env, PommermanEnv)
    assert env.observation_shape == (20, 11, 11)
