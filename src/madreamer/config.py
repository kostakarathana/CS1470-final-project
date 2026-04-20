from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvConfig:
    name: str = "mock_grid"
    num_agents: int = 2
    grid_size: int = 5
    max_steps: int = 16
    task_type: str = "cooperative"


@dataclass
class AlgorithmConfig:
    name: str = "shared"
    imagination_horizon: int = 5
    latent_dim: int = 64
    hidden_dim: int = 128
    encoder_channels: int = 32


@dataclass
class TrainingConfig:
    total_steps: int = 64
    batch_size: int = 16
    replay_capacity: int = 512
    learning_rate: float = 3e-4
    device: str = "cpu"


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


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    raw = _read_yaml(path)
    return ExperimentConfig(
        seed=raw.get("seed", 7),
        experiment_name=raw.get("experiment_name", "madreamer-smoke"),
        env=EnvConfig(**raw.get("env", {})),
        algorithm=AlgorithmConfig(**raw.get("algorithm", {})),
        training=TrainingConfig(**raw.get("training", {})),
    )
