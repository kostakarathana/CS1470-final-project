from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from madreamer.builders import ModuleBundle
from madreamer.config import ExperimentConfig
from madreamer.utils import ensure_dir


def save_checkpoint(
    path: str | Path,
    *,
    bundle: ModuleBundle,
    cfg: ExperimentConfig,
    step: int,
    metrics: dict[str, Any],
) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    torch.save(
        {
            "step": step,
            "config": cfg,
            "metrics": metrics,
            "world_models": {agent_id: model.state_dict() for agent_id, model in bundle.world_models.items()},
            "actors": {agent_id: model.state_dict() for agent_id, model in bundle.actors.items()},
            "critics": {agent_id: model.state_dict() for agent_id, model in bundle.critics.items()},
            "ppo_policies": {
                agent_id: model.state_dict() for agent_id, model in bundle.ppo_policies.items()
            },
        },
        target,
    )
    return target


def load_checkpoint(path: str | Path, bundle: ModuleBundle, device: str = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    for agent_id, state_dict in payload.get("world_models", {}).items():
        if agent_id in bundle.world_models:
            bundle.world_models[agent_id].load_state_dict(state_dict)
    for agent_id, state_dict in payload.get("actors", {}).items():
        if agent_id in bundle.actors:
            bundle.actors[agent_id].load_state_dict(state_dict)
    for agent_id, state_dict in payload.get("critics", {}).items():
        if agent_id in bundle.critics:
            bundle.critics[agent_id].load_state_dict(state_dict)
    for agent_id, state_dict in payload.get("ppo_policies", {}).items():
        if agent_id in bundle.ppo_policies:
            bundle.ppo_policies[agent_id].load_state_dict(state_dict)
    return payload
