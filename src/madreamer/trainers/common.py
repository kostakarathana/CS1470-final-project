from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingSummary:
    algorithm: str
    env_mode: str
    learner_setup: str
    total_env_steps: int
    episodes: int
    reward_totals: dict[str, float]
    replay_size: int
    latest_checkpoint_path: str | None = None
    latest_eval_metrics: dict[str, float] = field(default_factory=dict)


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
