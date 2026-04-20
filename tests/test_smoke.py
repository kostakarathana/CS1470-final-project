from pathlib import Path

import pytest

pytest.importorskip("torch")

from madreamer.config import load_experiment_config
from madreamer.experiment import run_experiment


def test_shared_smoke_run_collects_steps() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "shared.yaml"
    cfg = load_experiment_config(config_path)
    cfg.training.total_steps = 8
    summary = run_experiment(cfg)
    assert summary["collected_steps"] == 8
    assert summary["replay_size"] == 8


def test_ppo_smoke_run_collects_steps() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "ppo.yaml"
    cfg = load_experiment_config(config_path)
    cfg.training.total_steps = 8
    summary = run_experiment(cfg)
    assert summary["collected_steps"] == 8
    assert summary["strategy"] == "ppo"
