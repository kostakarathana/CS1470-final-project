from pathlib import Path

import pytest
import torch

from madreamer.builders import build_modules
from madreamer.config import load_experiment_config
from madreamer.envs.factory import build_env
from madreamer.models.policy import ActorNetwork
from madreamer.models.world_model import RSSMState, extract_observation_targets
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
    assert output.scalar_prediction.shape == (2, 6, 11, 11)
    assert imagined.reward_prediction.shape == (2,)
    env.close()


def test_world_model_targets_include_bombs_position_and_scalars() -> None:
    obs = torch.zeros((1, 20, 11, 11))
    obs[:, 10] = 1.0
    obs[:, 14, 2, 3] = 0.5
    obs[:, 15, 2, 3] = 0.7
    obs[:, 16, 4, 5] = 1.0
    obs[:, 17] = 0.2
    obs[:, 18] = 0.3
    obs[:, 19] = 1.0

    board_target, feature_target = extract_observation_targets(obs, board_value_count=14)

    assert board_target.shape == (1, 11, 11)
    assert feature_target.shape == (1, 6, 11, 11)
    assert feature_target[0, 0, 2, 3] == 0.5
    assert feature_target[0, 1, 2, 3] == 0.7
    assert feature_target[0, 2, 4, 5] == 1.0
    assert torch.all(feature_target[0, 3] == 0.2)
    assert torch.all(feature_target[0, 4] == 0.3)
    assert torch.all(feature_target[0, 5] == 1.0)


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
    assert "behavior_blocked_move_rate" in metrics
    assert "behavior_wasted_bomb_rate" in metrics
    env.close()


def test_dreamer_update_cadence_respects_train_every_steps(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "shared_smoke.yaml"
    cfg = load_experiment_config(config_path)
    cfg.algorithm.dreamer.train_every_steps = 4
    cfg.training.log_dir = str(tmp_path / "logs")
    cfg.training.checkpoint_dir = str(tmp_path / "checkpoints")
    env = build_env(cfg, pommerman_backend_factory=fake_backend_factory("ffa"))
    bundle = build_modules(cfg, env.agent_ids, env.observation_shape, env.action_dim, cfg.env.board_value_count)
    replay = MultiAgentReplayBuffer(capacity=128)
    trainer = DreamerCollector(env, bundle, cfg, replay)

    policies = {agent_id: (lambda _agent_id, _obs, _info: 0) for agent_id in env.agent_ids}
    collect_episode(env, policies, replay=replay, seed=0, episode_id=0)
    collect_episode(env, policies, replay=replay, seed=1, episode_id=1)

    trainer.env_steps = cfg.algorithm.dreamer.warmup_steps + 1
    assert not trainer._can_update_replay()

    trainer.env_steps = cfg.algorithm.dreamer.warmup_steps + cfg.algorithm.dreamer.train_every_steps
    assert trainer._can_update_replay()
    env.close()


def test_dreamer_advantage_normalization_preserves_raw_advantages() -> None:
    raw_advantages = [
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([4.0, 5.0, 6.0]),
    ]
    raw_snapshot = [advantage.clone() for advantage in raw_advantages]

    normalized = DreamerCollector._normalize_advantages(raw_advantages)
    normalized_flat = torch.cat([advantage.reshape(-1) for advantage in normalized])

    assert torch.allclose(normalized_flat.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(normalized_flat.std(unbiased=False), torch.tensor(1.0), atol=1e-6)
    for advantage, snapshot in zip(raw_advantages, raw_snapshot):
        assert torch.equal(advantage, snapshot)


def test_free_nats_applies_per_sample_before_averaging() -> None:
    kl_values = torch.tensor([0.0, 1.0])

    loss = DreamerCollector._apply_free_nats(kl_values, free_nats=0.5)

    assert torch.allclose(loss, torch.tensor(0.75))


def test_actor_network_respects_action_mask() -> None:
    actor = ActorNetwork(input_dim=3, hidden_dim=4, action_dim=3)
    for parameter in actor.parameters():
        parameter.data.zero_()
    actor.net[-1].bias.data = torch.tensor([0.0, 1.0, 2.0])
    features = torch.zeros(1, 3)

    output = actor.act(
        features,
        deterministic=True,
        action_mask=torch.tensor([[1, 1, 0]], dtype=torch.bool),
    )

    assert output.action.item() == 1
    assert output.logits[0, 2] < -1e30


def test_dreamer_imagined_action_mask_uses_decoded_board_and_position(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "shared_smoke.yaml"
    cfg = load_experiment_config(config_path)
    cfg.training.log_dir = str(tmp_path / "logs")
    cfg.training.checkpoint_dir = str(tmp_path / "checkpoints")
    env = build_env(cfg, pommerman_backend_factory=fake_backend_factory("ffa"))
    bundle = build_modules(cfg, env.agent_ids, env.observation_shape, env.action_dim, cfg.env.board_value_count)
    replay = MultiAgentReplayBuffer(capacity=128)
    trainer = DreamerCollector(env, bundle, cfg, replay)

    class DummyWorldModel:
        obs_shape = env.observation_shape
        board_value_count = cfg.env.board_value_count

        def decode(self, state: RSSMState):
            batch_size = state.deter.shape[0]
            board_logits = torch.zeros(batch_size, cfg.env.board_value_count, 11, 11)
            board_logits[:, 1, 1, 0] = 10.0
            feature_prediction = torch.zeros(batch_size, 6, 11, 11)
            feature_prediction[:, 2, 1, 1] = 1.0
            return (
                torch.zeros(batch_size),
                torch.zeros(batch_size),
                board_logits,
                feature_prediction,
            )

    state = RSSMState(
        deter=torch.zeros(1, cfg.algorithm.dreamer.hidden_dim),
        stoch=torch.zeros(1, cfg.algorithm.dreamer.latent_dim),
        mean=torch.zeros(1, cfg.algorithm.dreamer.latent_dim),
        std=torch.ones(1, cfg.algorithm.dreamer.latent_dim),
    )

    mask = trainer._imagined_action_mask(DummyWorldModel(), state)

    assert mask is not None
    assert mask[0, 0]
    assert mask[0, 2]
    assert not mask[0, 3]
    assert not mask[0, 5]
    env.close()


def test_dreamer_eval_comparison_prioritizes_win_rate(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "shared_smoke.yaml"
    cfg = load_experiment_config(config_path)
    cfg.training.log_dir = str(tmp_path / "logs")
    cfg.training.checkpoint_dir = str(tmp_path / "checkpoints")
    env = build_env(cfg, pommerman_backend_factory=fake_backend_factory("ffa"))
    bundle = build_modules(cfg, env.agent_ids, env.observation_shape, env.action_dim, cfg.env.board_value_count)
    replay = MultiAgentReplayBuffer(capacity=128)
    trainer = DreamerCollector(env, bundle, cfg, replay)

    incumbent = {"eval_win_rate": 0.25, "eval_mean_reward": 1.0}
    higher_reward_lower_win = {"eval_win_rate": 0.0, "eval_mean_reward": 10.0}
    same_win_higher_reward = {"eval_win_rate": 0.25, "eval_mean_reward": 1.5}

    assert not trainer._is_better_eval(higher_reward_lower_win, incumbent)
    assert trainer._is_better_eval(same_win_higher_reward, incumbent)
    env.close()


def test_dreamer_resume_restores_best_eval_metrics(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "shared_smoke.yaml"
    cfg = load_experiment_config(config_path)
    cfg.training.log_dir = str(tmp_path / "logs")
    cfg.training.checkpoint_dir = str(tmp_path / "checkpoints")
    env = build_env(cfg, pommerman_backend_factory=fake_backend_factory("ffa"))
    bundle = build_modules(cfg, env.agent_ids, env.observation_shape, env.action_dim, cfg.env.board_value_count)
    replay = MultiAgentReplayBuffer(capacity=128)
    trainer = DreamerCollector(env, bundle, cfg, replay)
    trainer.best_eval_metrics = {"eval_win_rate": 0.25, "eval_mean_reward": 0.5}
    trainer._save_checkpoint()
    checkpoint_path = trainer.checkpoint_path
    env.close()

    cfg.training.resume_checkpoint = str(checkpoint_path)
    resumed_env = build_env(cfg, pommerman_backend_factory=fake_backend_factory("ffa"))
    resumed_bundle = build_modules(
        cfg,
        resumed_env.agent_ids,
        resumed_env.observation_shape,
        resumed_env.action_dim,
        cfg.env.board_value_count,
    )
    resumed_replay = MultiAgentReplayBuffer(capacity=128)
    resumed = DreamerCollector(resumed_env, resumed_bundle, cfg, resumed_replay)

    assert resumed.best_eval_metrics == trainer.best_eval_metrics
    resumed_env.close()
