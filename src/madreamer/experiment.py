from __future__ import annotations

from dataclasses import asdict

from madreamer.builders import build_modules, move_bundle_to_device
from madreamer.config import ExperimentConfig
from madreamer.envs.factory import build_env
from madreamer.replay import MultiAgentReplayBuffer
from madreamer.trainers.dreamer import DreamerCollector
from madreamer.trainers.ppo import PPOCollector


def run_experiment(cfg: ExperimentConfig) -> dict[str, object]:
    env = build_env(cfg)
    bundle = build_modules(cfg, env.agent_ids, env.observation_shape, env.action_dim)
    move_bundle_to_device(bundle, cfg.training.device)
    replay = MultiAgentReplayBuffer(capacity=cfg.training.replay_capacity)

    if cfg.algorithm.name == "ppo":
        summary = PPOCollector(env, bundle, cfg, replay).run()
    else:
        summary = DreamerCollector(env, bundle, cfg, replay).run()

    payload = asdict(summary)
    payload["replay_size"] = len(replay)
    payload["experiment_name"] = cfg.experiment_name
    return payload
