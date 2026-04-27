from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
import time
from typing import TextIO


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


class TrainingProgress:
    def __init__(
        self,
        *,
        total_steps: int,
        label: str,
        stream: TextIO | None = None,
        width: int = 28,
        min_interval_seconds: float = 0.5,
    ) -> None:
        self.total_steps = max(1, int(total_steps))
        self.label = label
        self.stream = stream or sys.stderr
        self.width = width
        self.min_interval_seconds = min_interval_seconds
        self.start_time = time.monotonic()
        self.last_render_time = 0.0
        self.enabled = bool(getattr(self.stream, "isatty", lambda: False)())

    def update(
        self,
        current_steps: int,
        *,
        episodes: int,
        latest_eval_metrics: dict[str, float] | None = None,
        phase: str = "train",
        force: bool = False,
    ) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self.last_render_time < self.min_interval_seconds:
            return
        self.last_render_time = now
        self.stream.write(
            "\r" + self._render_line(current_steps, episodes, latest_eval_metrics or {}, phase)
        )
        self.stream.flush()

    def finish(
        self,
        current_steps: int,
        *,
        episodes: int,
        latest_eval_metrics: dict[str, float] | None = None,
    ) -> None:
        if not self.enabled:
            return
        self.stream.write(
            "\r" + self._render_line(current_steps, episodes, latest_eval_metrics or {}, "done") + "\n"
        )
        self.stream.flush()

    def _render_line(
        self,
        current_steps: int,
        episodes: int,
        latest_eval_metrics: dict[str, float],
        phase: str,
    ) -> str:
        current_steps = min(max(0, int(current_steps)), self.total_steps)
        fraction = current_steps / self.total_steps
        filled = int(round(self.width * fraction))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = max(0.0, time.monotonic() - self.start_time)
        rate = current_steps / elapsed if elapsed > 0.0 else 0.0
        remaining = self.total_steps - current_steps
        eta = remaining / rate if rate > 0.0 else None
        metrics = self._format_eval_metrics(latest_eval_metrics)
        return (
            f"{self.label} [{bar}] {fraction * 100:5.1f}% "
            f"{current_steps}/{self.total_steps} "
            f"eps={episodes} phase={phase} "
            f"elapsed={self._format_duration(elapsed)} eta={self._format_duration(eta)}"
            f"{metrics}"
        )

    def _format_eval_metrics(self, latest_eval_metrics: dict[str, float]) -> str:
        if not latest_eval_metrics:
            return ""
        reward = latest_eval_metrics.get("eval_mean_reward")
        win_rate = latest_eval_metrics.get("eval_win_rate")
        parts = []
        if reward is not None:
            parts.append(f"eval_reward={float(reward):.3f}")
        if win_rate is not None:
            parts.append(f"win={float(win_rate):.3f}")
        return " " + " ".join(parts) if parts else ""

    def _format_duration(self, seconds: float | None) -> str:
        if seconds is None:
            return "--:--"
        seconds = max(0, int(seconds))
        minutes, secs = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
