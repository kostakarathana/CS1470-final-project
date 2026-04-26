from __future__ import annotations

import json
from pathlib import Path

from analyze_results import curve_auc, load_metrics


def test_curve_auc_uses_trapezoids() -> None:
    assert curve_auc([0, 10, 20], [0.0, 1.0, 0.0]) == 10.0


def test_load_metrics_filters_non_eval_rows(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    rows = [
        {"phase": "train", "env_steps": 8, "world_model_loss": 1.0},
        {"phase": "eval", "env_steps": 16, "eval_mean_reward": 0.25, "eval_win_rate": 0.5},
    ]
    with (log_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    steps, rewards, win_rates, last_step, total_rows = load_metrics(log_dir)

    assert steps == [16]
    assert rewards == [0.25]
    assert win_rates == [0.5]
    assert last_step == 16
    assert total_rows == 2
