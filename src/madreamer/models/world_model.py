from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class WorldModelOutput:
    latent: Tensor
    hidden: Tensor
    reward_prediction: Tensor
    continuation_logit: Tensor
    reconstruction: Tensor


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, encoder_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, encoder_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(encoder_channels, encoder_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(encoder_channels, hidden_dim)

    def forward(self, obs: Tensor) -> Tensor:
        features = self.conv(obs).flatten(start_dim=1)
        return self.proj(features)


class ObservationDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, obs_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.obs_shape = obs_shape
        flat_dim = math.prod(obs_shape)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, flat_dim),
        )

    def forward(self, latent: Tensor) -> Tensor:
        reconstruction = self.net(latent)
        return reconstruction.view(latent.shape[0], *self.obs_shape)


class WorldModel(nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        action_dim: int,
        latent_dim: int,
        hidden_dim: int,
        encoder_channels: int,
        opponent_action_dim: int = 0,
    ) -> None:
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.opponent_action_dim = opponent_action_dim
        self.encoder = CNNEncoder(obs_shape[0], hidden_dim, encoder_channels)
        self.representation_dynamics = nn.GRUCell(
            hidden_dim + action_dim + opponent_action_dim,
            hidden_dim,
        )
        self.prior_dynamics = nn.GRUCell(action_dim + opponent_action_dim, hidden_dim)
        self.latent_head = nn.Linear(hidden_dim, latent_dim)
        self.reward_head = nn.Linear(latent_dim, 1)
        self.continuation_head = nn.Linear(latent_dim, 1)
        self.decoder = ObservationDecoder(latent_dim, hidden_dim, obs_shape)

    def initial_state(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def observe(
        self,
        obs: Tensor,
        prev_action: Tensor,
        prev_hidden: Tensor,
        opponent_action: Tensor | None = None,
    ) -> WorldModelOutput:
        encoded = self.encoder(obs.float())
        action_one_hot = self._action_one_hot(prev_action, obs.device)
        opponent_features = self._normalize_opponent_features(opponent_action, obs.shape[0], obs.device, obs.dtype)
        features = torch.cat([encoded, action_one_hot, opponent_features], dim=-1)
        hidden = self.representation_dynamics(features, prev_hidden)
        return self._build_output(hidden)

    def imagine(
        self,
        prev_hidden: Tensor,
        action: Tensor,
        opponent_action: Tensor | None = None,
    ) -> WorldModelOutput:
        action_one_hot = self._action_one_hot(action, prev_hidden.device)
        opponent_features = self._normalize_opponent_features(
            opponent_action,
            prev_hidden.shape[0],
            prev_hidden.device,
            prev_hidden.dtype,
        )
        hidden = self.prior_dynamics(torch.cat([action_one_hot, opponent_features], dim=-1), prev_hidden)
        return self._build_output(hidden)

    def forward(
        self,
        obs: Tensor,
        prev_action: Tensor,
        prev_hidden: Tensor,
        opponent_action: Tensor | None = None,
    ) -> WorldModelOutput:
        return self.observe(obs, prev_action, prev_hidden, opponent_action)

    def _build_output(self, hidden: Tensor) -> WorldModelOutput:
        latent = torch.tanh(self.latent_head(hidden))
        return WorldModelOutput(
            latent=latent,
            hidden=hidden,
            reward_prediction=self.reward_head(latent).squeeze(-1),
            continuation_logit=self.continuation_head(latent).squeeze(-1),
            reconstruction=self.decoder(latent),
        )

    def _action_one_hot(self, action: Tensor, device: torch.device) -> Tensor:
        return F.one_hot(action.long(), num_classes=self.action_dim).float().to(device)

    def _normalize_opponent_features(
        self,
        opponent_action: Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if not self.opponent_action_dim:
            return torch.zeros(batch_size, 0, device=device, dtype=dtype)
        if opponent_action is None:
            return torch.zeros(batch_size, self.opponent_action_dim, device=device, dtype=dtype)
        return opponent_action.to(device=device, dtype=dtype)
