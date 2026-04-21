from __future__ import annotations

from dataclasses import asdict

from madreamer.builders import build_modules, move_bundle_to_device
from madreamer.config import ExperimentConfig
from madreamer.envs.factory import build_env
from madreamer.replay import MultiAgentReplayBuffer
from madreamer.trainers.dreamer import DreamerCollector
from madreamer.trainers.ppo import PPOCollector


def run_experiment(
    cfg: ExperimentConfig,
    *,
    pommerman_backend_factory=None,
) -> dict[str, object]:
    env = build_env(cfg, pommerman_backend_factory=pommerman_backend_factory)
    bundle = build_modules(
        cfg,
        env.agent_ids,
        env.observation_shape,
        env.action_dim,
        cfg.env.board_value_count,
    )
    move_bundle_to_device(bundle, cfg.training.device)
    replay_capacity = (
        cfg.algorithm.dreamer.replay_capacity
        if cfg.algorithm.name != "ppo"
        else max(cfg.algorithm.ppo.rollout_steps * 8, 512)
    )
    replay = MultiAgentReplayBuffer(capacity=replay_capacity)

    if cfg.algorithm.name == "ppo":
        summary = PPOCollector(env, bundle, cfg, replay).run()
    else:
        summary = DreamerCollector(env, bundle, cfg, replay).run()

    payload = asdict(summary)
    payload["experiment_name"] = cfg.experiment_name
    env.close()
    return payload


def run_evaluation(
    cfg: ExperimentConfig,
    *,
    checkpoint_path: str | None = None,
    episodes: int | None = None,
    opponent_policy: str | None = None,
    pommerman_backend_factory=None,
) -> dict[str, object]:
    if checkpoint_path is not None:
        cfg.training.resume_checkpoint = checkpoint_path
    env = build_env(cfg, pommerman_backend_factory=pommerman_backend_factory)
    bundle = build_modules(
        cfg,
        env.agent_ids,
        env.observation_shape,
        env.action_dim,
        cfg.env.board_value_count,
    )
    move_bundle_to_device(bundle, cfg.training.device)
    replay_capacity = (
        cfg.algorithm.dreamer.replay_capacity
        if cfg.algorithm.name != "ppo"
        else max(cfg.algorithm.ppo.rollout_steps * 8, 512)
    )
    replay = MultiAgentReplayBuffer(capacity=replay_capacity)
    evaluator = PPOCollector(env, bundle, cfg, replay) if cfg.algorithm.name == "ppo" else DreamerCollector(env, bundle, cfg, replay)
    metrics = evaluator.evaluate(episodes or cfg.training.eval_episodes, opponent_policy=opponent_policy)
    env.close()
    return {
        "algorithm": cfg.algorithm.name,
        "env_mode": cfg.env.mode,
        "learner_setup": cfg.algorithm.learner_setup,
        "checkpoint_path": checkpoint_path,
        **metrics,
    }
