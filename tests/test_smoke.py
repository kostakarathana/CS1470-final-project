from pathlib import Path

import pytest

from madreamer.config import load_experiment_config
from madreamer.experiment import run_evaluation, run_experiment

from tests.helpers import fake_backend_factory


pytest.importorskip("torch")


def test_shared_smoke_run_collects_steps_with_fake_pommerman(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "shared_smoke.yaml"
    cfg = load_experiment_config(config_path)
    cfg.training.log_dir = str(tmp_path / "logs")
    cfg.training.checkpoint_dir = str(tmp_path / "checkpoints")
    summary = run_experiment(cfg, pommerman_backend_factory=fake_backend_factory("ffa"))
    assert summary["total_env_steps"] == cfg.training.total_steps
    assert summary["replay_size"] >= cfg.training.total_steps
    assert Path(summary["latest_checkpoint_path"]).exists()


def test_ppo_smoke_run_collects_steps(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "ppo_smoke.yaml"
    cfg = load_experiment_config(config_path)
    cfg.training.log_dir = str(tmp_path / "logs")
    cfg.training.checkpoint_dir = str(tmp_path / "checkpoints")
    summary = run_experiment(cfg)
    assert summary["total_env_steps"] == cfg.training.total_steps
    assert summary["algorithm"] == "ppo"


def test_team_eval_and_resume_roundtrip(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "team_shared_smoke.yaml"
    cfg = load_experiment_config(config_path)
    cfg.training.log_dir = str(tmp_path / "logs")
    cfg.training.checkpoint_dir = str(tmp_path / "checkpoints")
    summary = run_experiment(cfg, pommerman_backend_factory=fake_backend_factory("team"))
    metrics = run_evaluation(
        cfg,
        checkpoint_path=summary["latest_checkpoint_path"],
        episodes=2,
        pommerman_backend_factory=fake_backend_factory("team"),
    )
    assert metrics["env_mode"] == "team"
    assert metrics["eval_episodes"] == 2.0
