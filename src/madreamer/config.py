from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvConfig:
    name: str = "pommerman"
    mode: str = "ffa"
    env_id: str | None = None
    num_agents: int = 4
    grid_size: int = 5
    max_steps: int = 128
    task_type: str = "competitive"
    board_size: int = 11
    observability: str = "full"
    communication: bool = False
    board_value_count: int = 14


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    rollout_steps: int = 64
    minibatch_size: int = 64
    update_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5


@dataclass
class DreamerConfig:
    latent_dim: int = 32
    hidden_dim: int = 128
    encoder_channels: int = 32
    sequence_length: int = 8
    replay_capacity: int = 2048
    batch_size: int = 8
    warmup_steps: int = 64
    policy_warmup_steps: int = 0
    updates_per_collect: int = 2
    train_every_steps: int = 1
    model_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    critic_target_tau: float = 0.01
    imagination_horizon: int = 5
    gamma: float = 0.99
    lambda_return: float = 0.95
    kl_scale: float = 0.1
    free_nats: float = 0.5
    reward_scale: float = 1.0
    continuation_scale: float = 1.0
    reconstruction_scale: float = 1.0
    board_class_balance: float = 0.0
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0


@dataclass
class AlgorithmConfig:
    name: str = "ppo"
    learner_setup: str = "single_learner"
    opponent_policy: str = "simple"
    reward_preset: str = "sparse"
    ppo: PPOConfig = field(default_factory=PPOConfig)
    dreamer: DreamerConfig = field(default_factory=DreamerConfig)


@dataclass
class TrainingConfig:
    total_steps: int = 256
    device: str = "cpu"
    eval_interval_steps: int = 64
    eval_episodes: int = 4
    save_interval_steps: int = 64
    log_dir: str = "artifacts/logs"
    checkpoint_dir: str = "artifacts/checkpoints"
    resume_checkpoint: str | None = None


@dataclass
class ExperimentConfig:
    seed: int = 7
    experiment_name: str = "madreamer-smoke"
    env: EnvConfig = field(default_factory=EnvConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_env_config(raw: dict[str, Any]) -> EnvConfig:
    cfg = EnvConfig(**raw)
    if cfg.env_id is None:
        if cfg.mode == "ffa":
            cfg.env_id = "PommeFFACompetition-v0"
        elif cfg.mode == "team":
            cfg.env_id = "PommeTeamCompetition-v0"
        else:
            raise ValueError(f"Unsupported env.mode '{cfg.mode}'.")
    if cfg.mode == "team":
        cfg.task_type = "cooperative"
    return cfg


def _load_algorithm_config(raw: dict[str, Any]) -> AlgorithmConfig:
    top_level = {key: value for key, value in raw.items() if key not in {"ppo", "dreamer"}}
    cfg = AlgorithmConfig(**top_level)
    if "ppo" in raw:
        cfg.ppo = PPOConfig(**raw["ppo"])
    if "dreamer" in raw:
        cfg.dreamer = DreamerConfig(**raw["dreamer"])
    return cfg


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw = _read_yaml(path)
    return ExperimentConfig(
        seed=raw.get("seed", 7),
        experiment_name=raw.get("experiment_name", "madreamer-smoke"),
        env=_load_env_config(raw.get("env", {})),
        algorithm=_load_algorithm_config(raw.get("algorithm", {})),
        training=TrainingConfig(**raw.get("training", {})),
    )


def apply_overrides(
    cfg: ExperimentConfig,
    *,
    steps: int | None = None,
    seed: int | None = None,
    device: str | None = None,
    resume_checkpoint: str | None = None,
    log_dir: str | None = None,
) -> ExperimentConfig:
    if steps is not None:
        cfg.training.total_steps = steps
    if seed is not None:
        cfg.seed = seed
    if device is not None:
        cfg.training.device = device
    if resume_checkpoint is not None:
        cfg.training.resume_checkpoint = resume_checkpoint
    if log_dir is not None:
        cfg.training.log_dir = log_dir
    return cfg
