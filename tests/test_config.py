from pathlib import Path

from madreamer.config import load_experiment_config


def test_load_shared_config() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "shared.yaml"
    cfg = load_experiment_config(config_path)
    assert cfg.algorithm.name == "shared"
    assert cfg.env.num_agents == 2


def test_load_pommerman_phase1_config() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "pommerman_phase1.yaml"
    cfg = load_experiment_config(config_path)
    assert cfg.env.name == "pommerman"
    assert cfg.env.board_size == 11
    assert cfg.env.communication is False
