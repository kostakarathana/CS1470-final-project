from __future__ import annotations

from pathlib import Path

import pytest
import torch

from madreamer.builders import build_modules
from madreamer.config import ExperimentConfig
from diagnose_policy_behavior import _action_distribution
from visualize_game import _load_checkpoint_into_bundle


def test_visualization_checkpoint_loader_rejects_incompatible_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "empty.pt"
    torch.save({"bundle": {}}, checkpoint)
    cfg = ExperimentConfig()
    cfg.algorithm.name = "ppo"
    bundle = build_modules(cfg, ("agent_0",), (3, 5, 5), 5, 3)

    with pytest.raises(ValueError, match="compatible module weights"):
        _load_checkpoint_into_bundle(checkpoint, bundle, torch.device("cpu"))


def test_policy_diagnostic_action_distribution_includes_all_actions() -> None:
    distribution = _action_distribution({0: 2, 2: 1}, action_dim=4)

    assert distribution == {
        "action_0": 2 / 3,
        "action_1": 0.0,
        "action_2": 1 / 3,
        "action_3": 0.0,
    }
