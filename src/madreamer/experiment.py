from __future__ import annotations

from dataclasses import asdict

from madreamer.builders import build_modules, move_bundle_to_device
from madreamer.checkpoint import load_checkpoint
from madreamer.config import ExperimentConfig
from madreamer.envs.factory import build_env
from madreamer.replay import MultiAgentReplayBuffer
from madreamer.trainers.dreamer import DreamerTrainer
from madreamer.trainers.ppo import PPOTrainer


def run_experiment(cfg: ExperimentConfig, checkpoint_path: str | None = None) -> dict[str, object]:
    env = build_env(cfg)
    bundle = build_modules(cfg, env.agent_ids, env.observation_shape, env.action_dim)
    move_bundle_to_device(bundle, cfg.training.device)
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, bundle, device=cfg.training.device)
    replay = MultiAgentReplayBuffer(capacity=cfg.training.replay_capacity)

    if cfg.algorithm.name == "ppo":
        summary = PPOTrainer(env, bundle, cfg, replay).train()
    else:
        summary = DreamerTrainer(env, bundle, cfg, replay).train()

    payload = asdict(summary)
    payload["replay_size"] = len(replay)
    payload["experiment_name"] = cfg.experiment_name
    return payload
