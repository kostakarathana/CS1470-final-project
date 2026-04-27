from __future__ import annotations

from io import StringIO

from madreamer.trainers.common import TrainingProgress


class TtyStringIO(StringIO):
    def isatty(self) -> bool:
        return True


def test_training_progress_renders_eval_metrics() -> None:
    stream = TtyStringIO()
    progress = TrainingProgress(total_steps=100, label="demo", stream=stream)

    progress.update(
        25,
        episodes=3,
        latest_eval_metrics={"eval_mean_reward": 0.5, "eval_win_rate": 0.25},
        force=True,
    )
    progress.finish(
        100,
        episodes=8,
        latest_eval_metrics={"eval_mean_reward": 1.0, "eval_win_rate": 0.5},
    )

    rendered = stream.getvalue()
    assert "demo" in rendered
    assert "25/100" in rendered
    assert "100/100" in rendered
    assert "eval_reward=0.500" in rendered
    assert "win=0.500" in rendered
