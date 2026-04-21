from __future__ import annotations

from dataclasses import dataclass, field

from torch import nn

from madreamer.config import ExperimentConfig
from madreamer.models.policy import ActorNetwork, CriticNetwork, PPONetwork
from madreamer.models.world_model import WorldModel


@dataclass
class ModuleBundle:
    world_models: dict[str, WorldModel] = field(default_factory=dict)
    actors: dict[str, ActorNetwork] = field(default_factory=dict)
    critics: dict[str, CriticNetwork] = field(default_factory=dict)
    ppo_policies: dict[str, PPONetwork] = field(default_factory=dict)

    def unique_world_models(self) -> dict[str, WorldModel]:
        unique: dict[str, WorldModel] = {}
        seen: set[int] = set()
        for agent_id, model in self.world_models.items():
            if id(model) in seen:
                continue
            seen.add(id(model))
            unique[agent_id] = model
        return unique


def _build_world_model(
    cfg: ExperimentConfig,
    obs_shape: tuple[int, ...],
    action_dim: int,
    board_value_count: int,
    opponent_action_dim: int = 0,
) -> WorldModel:
    return WorldModel(
        obs_shape=obs_shape,
        action_dim=action_dim,
        latent_dim=cfg.algorithm.dreamer.latent_dim,
        hidden_dim=cfg.algorithm.dreamer.hidden_dim,
        encoder_channels=cfg.algorithm.dreamer.encoder_channels,
        board_value_count=board_value_count,
        opponent_action_dim=opponent_action_dim,
    )


def build_modules(
    cfg: ExperimentConfig,
    agent_ids: tuple[str, ...],
    obs_shape: tuple[int, ...],
    action_dim: int,
    board_value_count: int,
) -> ModuleBundle:
    strategy = cfg.algorithm.name
    bundle = ModuleBundle()
    hidden_dim = cfg.algorithm.dreamer.hidden_dim
    encoder_channels = cfg.algorithm.dreamer.encoder_channels

    if strategy == "ppo":
        for agent_id in agent_ids:
            bundle.ppo_policies[agent_id] = PPONetwork(
                obs_shape=obs_shape,
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                encoder_channels=encoder_channels,
            )
        return bundle

    if strategy == "shared":
        shared_model = _build_world_model(cfg, obs_shape, action_dim, board_value_count)
        for agent_id in agent_ids:
            bundle.world_models[agent_id] = shared_model
            bundle.actors[agent_id] = ActorNetwork(
                shared_model.features_dim,
                hidden_dim,
                action_dim,
            )
            bundle.critics[agent_id] = CriticNetwork(
                shared_model.features_dim,
                hidden_dim,
            )
        return bundle

    opponent_dim = (action_dim + 1) * (len(agent_ids) - 1) if strategy == "opponent_aware" else 0
    for agent_id in agent_ids:
        bundle.world_models[agent_id] = _build_world_model(
            cfg,
            obs_shape,
            action_dim,
            board_value_count,
            opponent_action_dim=opponent_dim,
        )
        bundle.actors[agent_id] = ActorNetwork(
            bundle.world_models[agent_id].features_dim,
            hidden_dim,
            action_dim,
        )
        bundle.critics[agent_id] = CriticNetwork(
            bundle.world_models[agent_id].features_dim,
            hidden_dim,
        )
    return bundle


def move_bundle_to_device(bundle: ModuleBundle, device: str) -> ModuleBundle:
    for module_group in (bundle.world_models, bundle.actors, bundle.critics, bundle.ppo_policies):
        for module in module_group.values():
            if isinstance(module, nn.Module):
                module.to(device)
    return bundle
