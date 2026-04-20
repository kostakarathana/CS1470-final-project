from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class RSSMState:
    deter: Tensor
    stoch: Tensor
    mean: Tensor
    std: Tensor

    @property
    def features(self) -> Tensor:
        return torch.cat([self.deter, self.stoch], dim=-1)


@dataclass
class WorldModelOutput:
    prior_state: RSSMState
    posterior_state: RSSMState
    reward_prediction: Tensor
    continuation_logit: Tensor
    board_logits: Tensor
    scalar_prediction: Tensor


@dataclass
class ImaginationOutput:
    next_state: RSSMState
    reward_prediction: Tensor
    continuation_logit: Tensor


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
        board_value_count: int,
        opponent_action_dim: int = 0,
    ) -> None:
        super().__init__()
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.board_value_count = board_value_count
        self.opponent_action_dim = opponent_action_dim
        self.encoder = CNNEncoder(obs_shape[0], hidden_dim, encoder_channels)
        self.rnn = nn.GRUCell(latent_dim + action_dim + opponent_action_dim, hidden_dim)
        self.prior_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.posterior_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        feature_dim = self.features_dim
        self.reward_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.continuation_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.board_decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, board_value_count * obs_shape[1] * obs_shape[2]),
        )
        self.scalar_decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    @property
    def features_dim(self) -> int:
        return self.hidden_dim + self.latent_dim

    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        zeros_deter = torch.zeros(batch_size, self.hidden_dim, device=device)
        zeros_stoch = torch.zeros(batch_size, self.latent_dim, device=device)
        ones_std = torch.ones(batch_size, self.latent_dim, device=device)
        return RSSMState(deter=zeros_deter, stoch=zeros_stoch, mean=zeros_stoch, std=ones_std)

    def observe(
        self,
        obs: Tensor,
        prev_action: Tensor,
        prev_state: RSSMState,
        opponent_action: Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> WorldModelOutput:
        prior_state = self.prior(prev_action, prev_state, opponent_action, deterministic=deterministic)
        encoded = self.encoder(obs.float())
        posterior_mean, posterior_std = self._stats(self.posterior_head(torch.cat([prior_state.deter, encoded], dim=-1)))
        posterior_stoch = self._sample(posterior_mean, posterior_std, deterministic=deterministic)
        posterior_state = RSSMState(
            deter=prior_state.deter,
            stoch=posterior_stoch,
            mean=posterior_mean,
            std=posterior_std,
        )
        reward_prediction, continuation_logit, board_logits, scalar_prediction = self.decode(posterior_state)
        return WorldModelOutput(
            prior_state=prior_state,
            posterior_state=posterior_state,
            reward_prediction=reward_prediction,
            continuation_logit=continuation_logit,
            board_logits=board_logits,
            scalar_prediction=scalar_prediction,
        )

    def prior(
        self,
        prev_action: Tensor,
        prev_state: RSSMState,
        opponent_action: Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> RSSMState:
        action_one_hot = F.one_hot(prev_action.long(), num_classes=self.action_dim).float()
        if self.opponent_action_dim:
            if opponent_action is None:
                opponent_action = torch.zeros(
                    prev_action.shape[0],
                    self.opponent_action_dim,
                    device=prev_action.device,
                    dtype=torch.float32,
                )
            transition_input = torch.cat([prev_state.stoch, action_one_hot, opponent_action], dim=-1)
        else:
            transition_input = torch.cat([prev_state.stoch, action_one_hot], dim=-1)
        deter = self.rnn(transition_input, prev_state.deter)
        mean, std = self._stats(self.prior_head(deter))
        stoch = self._sample(mean, std, deterministic=deterministic)
        return RSSMState(deter=deter, stoch=stoch, mean=mean, std=std)

    def imagine(
        self,
        prev_state: RSSMState,
        action: Tensor,
        opponent_action: Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> ImaginationOutput:
        next_state = self.prior(action, prev_state, opponent_action, deterministic=deterministic)
        reward_prediction, continuation_logit, _, _ = self.decode(next_state)
        return ImaginationOutput(
            next_state=next_state,
            reward_prediction=reward_prediction,
            continuation_logit=continuation_logit,
        )

    def decode(self, state: RSSMState) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        features = state.features
        reward_prediction = self.reward_head(features).squeeze(-1)
        continuation_logit = self.continuation_head(features).squeeze(-1)
        board_logits = self.board_decoder(features).reshape(
            features.shape[0],
            self.board_value_count,
            self.obs_shape[1],
            self.obs_shape[2],
        )
        scalar_prediction = self.scalar_decoder(features)
        return reward_prediction, continuation_logit, board_logits, scalar_prediction

    def _stats(self, stats: Tensor) -> tuple[Tensor, Tensor]:
        mean, std_param = stats.chunk(2, dim=-1)
        std = F.softplus(std_param) + 0.1
        return mean, std

    def _sample(self, mean: Tensor, std: Tensor, *, deterministic: bool) -> Tensor:
        if deterministic:
            return mean
        return mean + std * torch.randn_like(std)


def kl_divergence(posterior: RSSMState, prior: RSSMState) -> Tensor:
    var_post = posterior.std.square()
    var_prior = prior.std.square()
    return 0.5 * (
        2.0 * torch.log(prior.std) - 2.0 * torch.log(posterior.std)
        + (var_post + (posterior.mean - prior.mean).square()) / var_prior
        - 1.0
    ).sum(dim=-1)


def extract_observation_targets(obs: Tensor, board_value_count: int) -> tuple[Tensor, Tensor]:
    board_target = obs[:, :board_value_count].argmax(dim=1)
    scalar_target = torch.stack(
        [
            obs[:, board_value_count + 3, 0, 0],
            obs[:, board_value_count + 4, 0, 0],
            obs[:, board_value_count + 5, 0, 0],
        ],
        dim=-1,
    )
    return board_target.long(), scalar_target.float()
