from pathlib import Path

import pytest
import torch

from madreamer.builders import build_modules
from madreamer.config import load_experiment_config
from madreamer.envs.factory import build_env
from madreamer.replay import MultiAgentReplayBuffer
from madreamer.rollout import collect_episode
from madreamer.trainers.dreamer import DreamerCollector

from tests.helpers import fake_backend_factory


pytest.importorskip("torch")


def test_world_model_observe_and_imagine_shapes() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "shared_smoke.yaml"
    cfg = load_experiment_config(config_path)
    env = build_env(cfg, pommerman_backend_factory=fake_backend_factory("ffa"))
    bundle = build_modules(cfg, env.agent_ids, env.observation_shape, env.action_dim, cfg.env.board_value_count)
    model = bundle.world_models["agent_0"]
    obs = torch.zeros((2,) + env.observation_shape)
    prev_action = torch.zeros(2, dtype=torch.long)
    state = model.initial_state(2, torch.device("cpu"))
    output = model.observe(obs, prev_action, state)
    imagined = model.imagine(output.posterior_state, prev_action)

    assert output.posterior_state.features.shape[-1] == model.features_dim
    assert output.board_logits.shape == (2, cfg.env.board_value_count, 11, 11)
    assert imagined.reward_prediction.shape == (2,)
    env.close()


def test_dreamer_update_path_runs_on_fake_sequences(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "shared_smoke.yaml"
    cfg = load_experiment_config(config_path)
    cfg.training.log_dir = str(tmp_path / "logs")
    cfg.training.checkpoint_dir = str(tmp_path / "checkpoints")
    env = build_env(cfg, pommerman_backend_factory=fake_backend_factory("ffa"))
    bundle = build_modules(cfg, env.agent_ids, env.observation_shape, env.action_dim, cfg.env.board_value_count)
    replay = MultiAgentReplayBuffer(capacity=128)
    trainer = DreamerCollector(env, bundle, cfg, replay)

    policies = {agent_id: (lambda _agent_id, _obs, _info: 0) for agent_id in env.agent_ids}
    collect_episode(env, policies, replay=replay, seed=0, episode_id=0)
    collect_episode(env, policies, replay=replay, seed=1, episode_id=1)
    batch = replay.sample_sequences(2, cfg.algorithm.dreamer.sequence_length, env.agent_ids)
    world_metrics, start_states = trainer._update_world_models(batch)
    actor_metrics = trainer._update_actors_and_critics(batch, start_states)

    assert world_metrics["world_model_loss"] >= 0.0
    assert "world_model_board_accuracy" in world_metrics
    assert actor_metrics["critic_loss"] >= 0.0
    assert "imagined_action_entropy" in actor_metrics
    env.close()


def test_policy_warmup_skips_actor_critic_updates(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "shared_smoke.yaml"
    cfg = load_experiment_config(config_path)
    cfg.algorithm.dreamer.policy_warmup_steps = 100
    cfg.training.log_dir = str(tmp_path / "logs")
    cfg.training.checkpoint_dir = str(tmp_path / "checkpoints")
    env = build_env(cfg, pommerman_backend_factory=fake_backend_factory("ffa"))
    bundle = build_modules(cfg, env.agent_ids, env.observation_shape, env.action_dim, cfg.env.board_value_count)
    replay = MultiAgentReplayBuffer(capacity=128)
    trainer = DreamerCollector(env, bundle, cfg, replay)

    policies = {agent_id: (lambda _agent_id, _obs, _info: 0) for agent_id in env.agent_ids}
    collect_episode(env, policies, replay=replay, seed=0, episode_id=0)
    collect_episode(env, policies, replay=replay, seed=1, episode_id=1)
    trainer.env_steps = cfg.algorithm.dreamer.warmup_steps

    metrics = trainer._run_updates()

    assert metrics["policy_updates_enabled"] == 0.0
    assert metrics["policy_warmup_remaining"] > 0.0
    assert metrics["world_model_loss"] >= 0.0
    assert metrics["actor_loss"] == 0.0
    assert metrics["critic_loss"] == 0.0
    assert "behavior_action_0_rate" in metrics
    assert "behavior_safe_stop_rate" in metrics
    env.close()
