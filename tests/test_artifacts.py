from pathlib import Path

from madreamer.config import load_experiment_config
from madreamer.experiment import run_experiment


def test_ppo_run_writes_artifacts(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "ppo.yaml"
    cfg = load_experiment_config(config_path)
    cfg.training.total_steps = 8
    cfg.training.rollout_steps = 4
    cfg.training.eval_episodes = 1
    cfg.training.output_dir = str(tmp_path)
    summary = run_experiment(cfg)

    assert Path(summary["checkpoint_path"]).exists()
    assert Path(summary["metrics_path"]).exists()


def test_dreamer_run_writes_artifacts(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "shared.yaml"
    cfg = load_experiment_config(config_path)
    cfg.training.total_steps = 8
    cfg.training.rollout_steps = 4
    cfg.training.world_model_batch_size = 4
    cfg.training.world_model_updates = 1
    cfg.training.actor_value_updates = 1
    cfg.training.imagination_batch_size = 4
    cfg.training.eval_episodes = 1
    cfg.training.output_dir = str(tmp_path)
    summary = run_experiment(cfg)

    assert Path(summary["checkpoint_path"]).exists()
    assert Path(summary["metrics_path"]).exists()
