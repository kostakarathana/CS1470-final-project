from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from madreamer.models.world_model import CNNEncoder


class ActorNetwork(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, latent: Tensor) -> Tensor:
        return self.net(latent)

    def sample_action(self, latent: Tensor) -> Tensor:
        distribution = Categorical(logits=self(latent))
        return distribution.sample()


class CriticNetwork(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent: Tensor) -> Tensor:
        return self.net(latent).squeeze(-1)


@dataclass
class PPOPolicyOutput:
    action: Tensor
    value: Tensor


class PPONetwork(nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        hidden_dim: int,
        action_dim: int,
        encoder_channels: int,
    ) -> None:
        super().__init__()
        self.encoder = CNNEncoder(obs_shape[0], hidden_dim, encoder_channels)
        self.actor = ActorNetwork(hidden_dim, hidden_dim, action_dim)
        self.critic = CriticNetwork(hidden_dim, hidden_dim)

    def act(self, obs: Tensor) -> PPOPolicyOutput:
        encoded = self.encoder(obs.float())
        action = self.actor.sample_action(encoded)
        value = self.critic(encoded)
        return PPOPolicyOutput(action=action, value=value)
