from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(path: str | Path) -> list[dict[str, object]]:
    metrics_path = Path(path)
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    rows: list[dict[str, object]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plot_metrics(
    metrics_path: str | Path,
    output_path: str | Path,
    *,
    metric_key: str = "eval_mean_reward",
    x_key: str = "env_steps",
) -> Path:
    rows = [row for row in load_metrics(metrics_path) if metric_key in row and x_key in row]
    if not rows:
        raise ValueError(f"No metric rows found for '{metric_key}'.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot([row[x_key] for row in rows], [row[metric_key] for row in rows], marker="o")
    plt.xlabel(x_key.replace("_", " ").title())
    plt.ylabel(metric_key.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    return output
