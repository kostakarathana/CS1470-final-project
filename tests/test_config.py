from pathlib import Path

from madreamer.config import load_experiment_config


def test_load_ppo_smoke_config() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "ppo_smoke.yaml"
    cfg = load_experiment_config(config_path)
    assert cfg.algorithm.name == "ppo"
    assert cfg.algorithm.ppo.rollout_steps == 8
    assert cfg.env.name == "mock_grid"


def test_load_team_shared_smoke_config() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "team_shared_smoke.yaml"
    cfg = load_experiment_config(config_path)
    assert cfg.env.mode == "team"
    assert cfg.algorithm.learner_setup == "multi_learner"
    assert cfg.algorithm.reward_preset == "shaped"
