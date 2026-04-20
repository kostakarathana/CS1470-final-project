from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical
import torch.nn.functional as F

from madreamer.builders import ModuleBundle
from madreamer.checkpoint import save_checkpoint
from madreamer.config import ExperimentConfig
from madreamer.envs.base import MultiAgentEnv
from madreamer.replay import MultiAgentReplayBuffer, Transition
from madreamer.scripted import build_scripted_policy
from madreamer.utils import ensure_dir, save_json


@dataclass
class TrainerSummary:
    collected_steps: int
    episodes: int
    reward_totals: dict[str, float]
    strategy: str
    checkpoint_path: str
    metrics_path: str
    final_eval_reward_mean: float
    final_eval_win_rate: float


class DreamerTrainer:
    def __init__(
        self,
        env: MultiAgentEnv,
        bundle: ModuleBundle,
        cfg: ExperimentConfig,
        replay: MultiAgentReplayBuffer,
    ) -> None:
        self.env = env
        self.bundle = bundle
        self.cfg = cfg
        self.replay = replay
        self.device = torch.device(cfg.training.device)
        self.trainable_agent_ids = (
            (self.env.agent_ids[0],) if cfg.env.opponent_mode == "fixed" else self.env.agent_ids
        )
        self.scripted_policies = {
            agent_id: build_scripted_policy(env, cfg.seed + index, env.action_dim)
            for index, agent_id in enumerate(self.env.agent_ids)
            if agent_id not in self.trainable_agent_ids
        }
        self.actor_optimizers = {
            agent_id: torch.optim.Adam(self.bundle.actors[agent_id].parameters(), lr=cfg.training.learning_rate)
            for agent_id in self.trainable_agent_ids
        }
        self.critic_optimizers = {
            agent_id: torch.optim.Adam(self.bundle.critics[agent_id].parameters(), lr=cfg.training.learning_rate)
            for agent_id in self.trainable_agent_ids
        }
        self.world_model_agents = _group_agents_by_world_model(self.bundle, self.trainable_agent_ids)
        self.world_model_optimizers = {
            key: torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
            for key, (model, _) in self.world_model_agents.items()
        }
        self.output_dir = ensure_dir(Path(cfg.training.output_dir) / cfg.experiment_name)
        self.metrics_history: list[dict[str, Any]] = []

    def train(self) -> TrainerSummary:
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        observations = self.env.reset(seed=self.cfg.seed)
        infos = {agent_id: {} for agent_id in self.env.agent_ids}
        hidden = {
            agent_id: self.bundle.world_models[agent_id].initial_state(1, self.device)
            for agent_id in self.trainable_agent_ids
        }
        prev_actions = {agent_id: 0 for agent_id in self.env.agent_ids}
        global_step = 0
        episodes = 0
        reward_totals = {agent_id: 0.0 for agent_id in self.env.agent_ids}
        best_eval = float("-inf")
        checkpoint_path = self.output_dir / "latest.pt"

        while global_step < self.cfg.training.total_steps:
            steps_this_round = 0
            while (
                steps_this_round < self.cfg.training.rollout_steps
                and global_step < self.cfg.training.total_steps
            ):
                actions: dict[str, int] = {}
                with torch.no_grad():
                    for agent_id in self.env.agent_ids:
                        if agent_id in self.trainable_agent_ids:
                            obs_tensor = torch.tensor(observations[agent_id][None], device=self.device)
                            prev_action_tensor = torch.tensor([prev_actions[agent_id]], device=self.device)
                            opponent_tensor = self._build_real_opponent_tensor(agent_id, prev_actions)
                            observed = self.bundle.world_models[agent_id].observe(
                                obs_tensor,
                                prev_action_tensor,
                                hidden[agent_id],
                                opponent_action=opponent_tensor,
                            )
                            hidden[agent_id] = observed.hidden
                            logits = self.bundle.actors[agent_id](observed.latent)
                            actions[agent_id] = int(Categorical(logits=logits).sample().item())
                        else:
                            actions[agent_id] = self.scripted_policies[agent_id](
                                agent_id,
                                observations[agent_id],
                                infos.get(agent_id, {}),
                            )

                step = self.env.step(actions)
                self.replay.add(
                    Transition(
                        observations={agent_id: obs.copy() for agent_id, obs in observations.items()},
                        actions=actions.copy(),
                        rewards=step.rewards.copy(),
                        next_observations={agent_id: obs.copy() for agent_id, obs in step.observations.items()},
                        terminated=step.terminated.copy(),
                        truncated=step.truncated.copy(),
                        infos={agent_id: dict(info) for agent_id, info in step.infos.items()},
                    )
                )
                for agent_id, reward in step.rewards.items():
                    reward_totals[agent_id] += reward
                observations = step.observations
                infos = step.infos
                prev_actions = actions
                steps_this_round += 1
                global_step += 1
                if step.done:
                    episodes += 1
                    observations = self.env.reset(seed=self.cfg.seed + episodes + global_step)
                    infos = {agent_id: {} for agent_id in self.env.agent_ids}
                    hidden = {
                        agent_id: self.bundle.world_models[agent_id].initial_state(1, self.device)
                        for agent_id in self.trainable_agent_ids
                    }
                    prev_actions = {agent_id: 0 for agent_id in self.env.agent_ids}

            metrics = self._update_models() if len(self.replay) >= self.cfg.training.world_model_batch_size else {}
            eval_metrics = self.evaluate(self.cfg.training.eval_episodes)
            log_record = {
                "step": global_step,
                "episodes": episodes,
                "world_model_loss": metrics.get("world_model_loss", 0.0),
                "actor_loss": metrics.get("actor_loss", 0.0),
                "critic_loss": metrics.get("critic_loss", 0.0),
                "eval_reward_mean": eval_metrics["mean_reward"],
                "eval_win_rate": eval_metrics["win_rate"],
            }
            self.metrics_history.append(log_record)
            if eval_metrics["mean_reward"] >= best_eval:
                best_eval = eval_metrics["mean_reward"]
                checkpoint_path = save_checkpoint(
                    self.output_dir / "best.pt",
                    bundle=self.bundle,
                    cfg=self.cfg,
                    step=global_step,
                    metrics=log_record,
                )
            save_checkpoint(
                self.output_dir / "latest.pt",
                bundle=self.bundle,
                cfg=self.cfg,
                step=global_step,
                metrics=log_record,
            )

        metrics_path = save_json(self.output_dir / "metrics.json", self.metrics_history)
        return TrainerSummary(
            collected_steps=global_step,
            episodes=episodes,
            reward_totals=reward_totals,
            strategy=self.cfg.algorithm.name,
            checkpoint_path=str(checkpoint_path),
            metrics_path=str(metrics_path),
            final_eval_reward_mean=self.metrics_history[-1]["eval_reward_mean"] if self.metrics_history else 0.0,
            final_eval_win_rate=self.metrics_history[-1]["eval_win_rate"] if self.metrics_history else 0.0,
        )

    def evaluate(self, num_episodes: int) -> dict[str, float]:
        reward_sums: list[float] = []
        wins = 0.0
        for offset in range(num_episodes):
            observations = self.env.reset(seed=self.cfg.seed + 20_000 + offset)
            infos = {agent_id: {} for agent_id in self.env.agent_ids}
            hidden = {
                agent_id: self.bundle.world_models[agent_id].initial_state(1, self.device)
                for agent_id in self.trainable_agent_ids
            }
            prev_actions = {agent_id: 0 for agent_id in self.env.agent_ids}
            done = False
            reward_total = 0.0
            while not done:
                actions: dict[str, int] = {}
                with torch.no_grad():
                    for agent_id in self.env.agent_ids:
                        if agent_id in self.trainable_agent_ids:
                            obs_tensor = torch.tensor(observations[agent_id][None], device=self.device)
                            prev_action_tensor = torch.tensor([prev_actions[agent_id]], device=self.device)
                            opponent_tensor = self._build_real_opponent_tensor(agent_id, prev_actions)
                            observed = self.bundle.world_models[agent_id].observe(
                                obs_tensor,
                                prev_action_tensor,
                                hidden[agent_id],
                                opponent_action=opponent_tensor,
                            )
                            hidden[agent_id] = observed.hidden
                            actions[agent_id] = int(
                                self.bundle.actors[agent_id](observed.latent).argmax(dim=-1).item()
                            )
                        else:
                            actions[agent_id] = self.scripted_policies[agent_id](
                                agent_id,
                                observations[agent_id],
                                infos.get(agent_id, {}),
                            )
                step = self.env.step(actions)
                reward_total += step.rewards[self.trainable_agent_ids[0]]
                observations = step.observations
                infos = step.infos
                prev_actions = actions
                done = step.done
            reward_sums.append(reward_total)
            if reward_total > 0:
                wins += 1.0
        return {
            "mean_reward": float(np.mean(reward_sums) if reward_sums else 0.0),
            "win_rate": wins / max(1, num_episodes),
        }

    def _update_models(self) -> dict[str, float]:
        world_model_losses = []
        actor_losses = []
        critic_losses = []
        for _ in range(self.cfg.training.world_model_updates):
            batch = self.replay.sample(self.cfg.training.world_model_batch_size)
            for world_model_key, (world_model, agent_ids) in self.world_model_agents.items():
                optimizer = self.world_model_optimizers[world_model_key]
                losses = []
                for agent_id in agent_ids:
                    obs, next_obs, actions, rewards, dones, opponent_tensor = self._prepare_transition_batch(
                        batch,
                        agent_id,
                    )
                    zero_action = torch.zeros(actions.shape[0], device=self.device, dtype=torch.long)
                    zero_hidden = world_model.initial_state(actions.shape[0], self.device)
                    current = world_model.observe(obs, zero_action, zero_hidden)
                    predicted = world_model.imagine(current.hidden, actions, opponent_tensor)
                    target_continue = 1.0 - dones
                    loss = (
                        F.mse_loss(predicted.reconstruction, next_obs)
                        + F.mse_loss(predicted.reward_prediction, rewards)
                        + F.binary_cross_entropy_with_logits(
                            predicted.continuation_logit,
                            target_continue,
                        )
                    )
                    losses.append(loss)
                total_loss = torch.stack(losses).mean()
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(world_model.parameters(), self.cfg.training.max_grad_norm)
                optimizer.step()
                world_model_losses.append(float(total_loss.item()))

        for _ in range(self.cfg.training.actor_value_updates):
            batch = self.replay.sample(self.cfg.training.imagination_batch_size)
            for agent_id in self.trainable_agent_ids:
                obs, _next_obs, _actions, _rewards, _dones, _opponent_tensor = self._prepare_transition_batch(
                    batch,
                    agent_id,
                )
                world_model = self.bundle.world_models[agent_id]
                zero_action = torch.zeros(obs.shape[0], device=self.device, dtype=torch.long)
                zero_hidden = world_model.initial_state(obs.shape[0], self.device)
                current = world_model.observe(obs, zero_action, zero_hidden)
                hidden = current.hidden
                latents = []
                log_probs = []
                entropies = []
                rewards = []
                continuations = []
                values = []
                for _ in range(self.cfg.algorithm.imagination_horizon):
                    latent = world_model._build_output(hidden).latent
                    logits = self.bundle.actors[agent_id](latent)
                    distribution = Categorical(logits=logits)
                    action = distribution.sample()
                    latents.append(latent)
                    log_probs.append(distribution.log_prob(action))
                    entropies.append(distribution.entropy())
                    values.append(self.bundle.critics[agent_id](latent))
                    imagined = world_model.imagine(hidden, action)
                    rewards.append(imagined.reward_prediction)
                    continuations.append(torch.sigmoid(imagined.continuation_logit))
                    hidden = imagined.hidden
                bootstrap_value = self.bundle.critics[agent_id](world_model._build_output(hidden).latent).detach()
                returns = _lambda_returns(
                    rewards=rewards,
                    continuations=continuations,
                    values=values,
                    bootstrap_value=bootstrap_value,
                    gamma=self.cfg.training.gamma,
                    gae_lambda=self.cfg.training.gae_lambda,
                )
                value_tensor = torch.stack(values)
                log_prob_tensor = torch.stack(log_probs)
                entropy_tensor = torch.stack(entropies)
                advantages = returns.detach() - value_tensor
                actor_loss = -(log_prob_tensor * advantages.detach()).mean() - (
                    self.cfg.training.entropy_coef * entropy_tensor.mean()
                )
                critic_loss = F.mse_loss(value_tensor, returns.detach())

                self.actor_optimizers[agent_id].zero_grad()
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(
                    self.bundle.actors[agent_id].parameters(),
                    self.cfg.training.max_grad_norm,
                )
                self.actor_optimizers[agent_id].step()

                self.critic_optimizers[agent_id].zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.bundle.critics[agent_id].parameters(),
                    self.cfg.training.max_grad_norm,
                )
                self.critic_optimizers[agent_id].step()
                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))

        return {
            "world_model_loss": float(np.mean(world_model_losses) if world_model_losses else 0.0),
            "actor_loss": float(np.mean(actor_losses) if actor_losses else 0.0),
            "critic_loss": float(np.mean(critic_losses) if critic_losses else 0.0),
        }

    def _prepare_transition_batch(
        self,
        batch: list[Transition],
        agent_id: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None]:
        obs = torch.tensor(
            np.stack([transition.observations[agent_id] for transition in batch]),
            device=self.device,
        ).float()
        next_obs = torch.tensor(
            np.stack([transition.next_observations[agent_id] for transition in batch]),
            device=self.device,
        ).float()
        actions = torch.tensor(
            [transition.actions[agent_id] for transition in batch],
            device=self.device,
            dtype=torch.long,
        )
        rewards = torch.tensor(
            [transition.rewards[agent_id] for transition in batch],
            device=self.device,
            dtype=torch.float32,
        )
        dones = torch.tensor(
            [
                float(transition.terminated[agent_id] or transition.truncated[agent_id])
                for transition in batch
            ],
            device=self.device,
            dtype=torch.float32,
        )
        opponent_tensor = self._build_batch_opponent_tensor(agent_id, batch)
        return obs, next_obs, actions, rewards, dones, opponent_tensor

    def _build_batch_opponent_tensor(
        self,
        agent_id: str,
        batch: list[Transition],
    ) -> Tensor | None:
        world_model = self.bundle.world_models[agent_id]
        if not world_model.opponent_action_dim:
            return None
        opponent_ids = [other_id for other_id in self.env.agent_ids if other_id != agent_id]
        tensors = []
        for transition in batch:
            one_hots = []
            for other_id in opponent_ids:
                action = torch.tensor(transition.actions[other_id], device=self.device)
                one_hots.append(F.one_hot(action, num_classes=self.env.action_dim).float())
            tensors.append(torch.cat(one_hots, dim=0))
        return torch.stack(tensors, dim=0)

    def _build_real_opponent_tensor(
        self,
        agent_id: str,
        previous_actions: dict[str, int],
    ) -> Tensor | None:
        world_model = self.bundle.world_models[agent_id]
        if not world_model.opponent_action_dim:
            return None
        opponent_ids = [other_id for other_id in self.env.agent_ids if other_id != agent_id]
        one_hots = []
        for other_id in opponent_ids:
            action = torch.tensor(previous_actions[other_id], device=self.device)
            one_hots.append(F.one_hot(action, num_classes=self.env.action_dim).float())
        return torch.cat(one_hots, dim=0).unsqueeze(0)


def _group_agents_by_world_model(
    bundle: ModuleBundle,
    trainable_agent_ids: tuple[str, ...],
) -> dict[int, tuple[torch.nn.Module, list[str]]]:
    grouped: dict[int, tuple[torch.nn.Module, list[str]]] = {}
    for agent_id in trainable_agent_ids:
        model = bundle.world_models[agent_id]
        key = id(model)
        if key not in grouped:
            grouped[key] = (model, [])
        grouped[key][1].append(agent_id)
    return grouped


def _lambda_returns(
    rewards: list[Tensor],
    continuations: list[Tensor],
    values: list[Tensor],
    bootstrap_value: Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tensor:
    returns = []
    next_return = bootstrap_value
    next_value = bootstrap_value
    for reward, continuation, value in zip(
        reversed(rewards),
        reversed(continuations),
        reversed(values),
    ):
        target = reward + gamma * continuation * ((1 - gae_lambda) * next_value + gae_lambda * next_return)
        returns.append(target)
        next_return = target
        next_value = value.detach()
    return torch.stack(list(reversed(returns)), dim=0)
