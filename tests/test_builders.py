import pytest

pytest.importorskip("torch")

from madreamer.builders import build_modules
from madreamer.config import ExperimentConfig


def test_shared_strategy_reuses_world_model() -> None:
    cfg = ExperimentConfig()
    cfg.algorithm.name = "shared"
    bundle = build_modules(cfg, ("agent_0", "agent_1", "agent_2", "agent_3"), (20, 11, 11), 6, 14)
    assert bundle.world_models["agent_0"] is bundle.world_models["agent_1"]


def test_independent_strategy_keeps_separate_world_models() -> None:
    cfg = ExperimentConfig()
    cfg.algorithm.name = "independent"
    bundle = build_modules(cfg, ("agent_0", "agent_1", "agent_2", "agent_3"), (20, 11, 11), 6, 14)
    assert bundle.world_models["agent_0"] is not bundle.world_models["agent_1"]


def test_opponent_aware_world_model_tracks_opponent_action_dim() -> None:
    cfg = ExperimentConfig()
    cfg.algorithm.name = "opponent_aware"
    bundle = build_modules(cfg, ("agent_0", "agent_1", "agent_2", "agent_3"), (20, 11, 11), 6, 14)
    assert bundle.world_models["agent_0"].opponent_action_dim == 21
