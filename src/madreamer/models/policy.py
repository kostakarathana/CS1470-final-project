from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn
from torch.distributions import Categorical

from madreamer.models.world_model import CNNEncoder


@dataclass
class ActorOutput:
    action: Tensor
    log_prob: Tensor
    entropy: Tensor
    logits: Tensor


@dataclass
class PPOPolicyOutput:
    action: Tensor
    log_prob: Tensor
    value: Tensor
    entropy: Tensor


class ActorNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)

    def distribution(self, features: Tensor) -> Categorical:
        return Categorical(logits=self(features))

    def act(self, features: Tensor, *, deterministic: bool = False) -> ActorOutput:
        logits = self(features)
        distribution = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return ActorOutput(
            action=action,
            log_prob=distribution.log_prob(action),
            entropy=distribution.entropy(),
            logits=logits,
        )


class CriticNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features).squeeze(-1)


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

    def encode(self, obs: Tensor) -> Tensor:
        return self.encoder(obs.float())

    def act(self, obs: Tensor, *, deterministic: bool = False) -> PPOPolicyOutput:
        encoded = self.encode(obs)
        actor_output = self.actor.act(encoded, deterministic=deterministic)
        value = self.critic(encoded)
        return PPOPolicyOutput(
            action=actor_output.action,
            log_prob=actor_output.log_prob,
            value=value,
            entropy=actor_output.entropy,
        )

    def evaluate_actions(self, obs: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        encoded = self.encode(obs)
        distribution = self.actor.distribution(encoded)
        log_prob = distribution.log_prob(actions.long())
        entropy = distribution.entropy()
        value = self.critic(encoded)
        return log_prob, entropy, value
