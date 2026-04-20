from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class WorldModelOutput:
    latent: Tensor
    hidden: Tensor
    reward_prediction: Tensor


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
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.opponent_action_dim = opponent_action_dim
        self.encoder = CNNEncoder(obs_shape[0], hidden_dim, encoder_channels)
        input_dim = hidden_dim + action_dim + opponent_action_dim
        self.dynamics = nn.GRUCell(input_dim, hidden_dim)
        self.latent_head = nn.Linear(hidden_dim, latent_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)

    def initial_state(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(
        self,
        obs: Tensor,
        prev_action: Tensor,
        prev_hidden: Tensor,
        opponent_action: Tensor | None = None,
    ) -> WorldModelOutput:
        encoded = self.encoder(obs.float())
        action_one_hot = F.one_hot(prev_action.long(), num_classes=self.action_dim).float()
        if self.opponent_action_dim:
            if opponent_action is None:
                opponent_action = torch.zeros(
                    obs.shape[0],
                    self.opponent_action_dim,
                    device=obs.device,
                    dtype=obs.dtype,
                )
            features = torch.cat([encoded, action_one_hot, opponent_action], dim=-1)
        else:
            features = torch.cat([encoded, action_one_hot], dim=-1)
        hidden = self.dynamics(features, prev_hidden)
        latent = torch.tanh(self.latent_head(hidden))
        reward_prediction = self.reward_head(hidden).squeeze(-1)
        return WorldModelOutput(latent=latent, hidden=hidden, reward_prediction=reward_prediction)
