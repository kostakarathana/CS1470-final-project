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


def _build_world_model(
    cfg: ExperimentConfig,
    obs_shape: tuple[int, ...],
    action_dim: int,
    opponent_action_dim: int = 0,
) -> WorldModel:
    return WorldModel(
        obs_shape=obs_shape,
        action_dim=action_dim,
        latent_dim=cfg.algorithm.latent_dim,
        hidden_dim=cfg.algorithm.hidden_dim,
        encoder_channels=cfg.algorithm.encoder_channels,
        opponent_action_dim=opponent_action_dim,
    )


def build_modules(
    cfg: ExperimentConfig,
    agent_ids: tuple[str, ...],
    obs_shape: tuple[int, ...],
    action_dim: int,
) -> ModuleBundle:
    strategy = cfg.algorithm.name
    bundle = ModuleBundle()

    if strategy == "ppo":
        for agent_id in agent_ids:
            bundle.ppo_policies[agent_id] = PPONetwork(
                obs_shape=obs_shape,
                hidden_dim=cfg.algorithm.hidden_dim,
                action_dim=action_dim,
                encoder_channels=cfg.algorithm.encoder_channels,
            )
        return bundle

    if strategy == "shared":
        shared_model = _build_world_model(cfg, obs_shape, action_dim)
        for agent_id in agent_ids:
            bundle.world_models[agent_id] = shared_model
            bundle.actors[agent_id] = ActorNetwork(
                cfg.algorithm.latent_dim,
                cfg.algorithm.hidden_dim,
                action_dim,
            )
            bundle.critics[agent_id] = CriticNetwork(
                cfg.algorithm.latent_dim,
                cfg.algorithm.hidden_dim,
            )
        return bundle

    opponent_dim = action_dim * (len(agent_ids) - 1) if strategy == "opponent_aware" else 0
    for agent_id in agent_ids:
        bundle.world_models[agent_id] = _build_world_model(
            cfg,
            obs_shape,
            action_dim,
            opponent_action_dim=opponent_dim,
        )
        bundle.actors[agent_id] = ActorNetwork(
            cfg.algorithm.latent_dim,
            cfg.algorithm.hidden_dim,
            action_dim,
        )
        bundle.critics[agent_id] = CriticNetwork(
            cfg.algorithm.latent_dim,
            cfg.algorithm.hidden_dim,
        )
    return bundle


def move_bundle_to_device(bundle: ModuleBundle, device: str) -> ModuleBundle:
    for module_group in (bundle.world_models, bundle.actors, bundle.critics, bundle.ppo_policies):
        for module in module_group.values():
            if isinstance(module, nn.Module):
                module.to(device)
    return bundle
