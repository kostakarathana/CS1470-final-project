from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from madreamer.builders import ModuleBundle
from madreamer.config import ExperimentConfig
from madreamer.envs.base import MultiAgentEnv
from madreamer.models.world_model import RSSMState, extract_observation_targets, kl_divergence
from madreamer.opponents import FixedOpponentManager
from madreamer.replay import MultiAgentReplayBuffer, ReplayStep, build_opponent_context
from madreamer.tracking import JsonlLogger
from madreamer.trainers.common import TrainingSummary, ensure_dir


class DreamerCollector:
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
        self.controlled_agent_ids = (
            (env.agent_ids[0],)
            if cfg.algorithm.learner_setup == "single_learner"
            else env.agent_ids
        )
        self.opponents = FixedOpponentManager(
            env,
            policy_name=cfg.algorithm.opponent_policy,
            controlled_agent_ids=self.controlled_agent_ids,
            seed=cfg.seed,
        )
        self.logger = JsonlLogger(cfg.training.log_dir)
        checkpoint_dir = ensure_dir(cfg.training.checkpoint_dir)
        self.checkpoint_path = checkpoint_dir / f"{cfg.experiment_name}_{cfg.algorithm.name}_latest.pt"
        self.reward_totals = {agent_id: 0.0 for agent_id in self.env.agent_ids}
        self.env_steps = 0
        self.episodes = 0
        self.episode_id = 0
        self.latest_eval_metrics: dict[str, float] = {}
        self.world_optimizers = {
            agent_id: torch.optim.Adam(
                model.parameters(),
                lr=cfg.algorithm.dreamer.model_learning_rate,
            )
            for agent_id, model in self.bundle.unique_world_models().items()
        }
        self.actor_optimizers = {
            agent_id: torch.optim.Adam(
                self.bundle.actors[agent_id].parameters(),
                lr=cfg.algorithm.dreamer.actor_learning_rate,
            )
            for agent_id in self.controlled_agent_ids
        }
        self.critic_optimizers = {
            agent_id: torch.optim.Adam(
                self.bundle.critics[agent_id].parameters(),
                lr=cfg.algorithm.dreamer.critic_learning_rate,
            )
            for agent_id in self.controlled_agent_ids
        }
        if cfg.training.resume_checkpoint:
            self._load_checkpoint(Path(cfg.training.resume_checkpoint))

    def run(self) -> TrainingSummary:
        observations = self.env.reset(seed=self.cfg.seed)
        infos = self.env.last_infos
        states, prev_actions, prev_opponent_contexts = self._initial_rollout_state()
        next_eval_at = self.cfg.training.eval_interval_steps
        next_save_at = self.cfg.training.save_interval_steps

        while self.env_steps < self.cfg.training.total_steps:
            actions = self.opponents.actions(observations, infos)
            for agent_id in self.controlled_agent_ids:
                world_model = self.bundle.world_models[agent_id]
                obs_tensor = self._obs_tensor(observations[agent_id][None])
                prev_action = torch.as_tensor([prev_actions[agent_id]], device=self.device, dtype=torch.long)
                opponent_context = prev_opponent_contexts[agent_id]
                opponent_tensor = (
                    torch.as_tensor(opponent_context[None], device=self.device, dtype=torch.float32)
                    if world_model.opponent_action_dim
                    else None
                )
                with torch.no_grad():
                    model_output = world_model.observe(
                        obs_tensor,
                        prev_action,
                        states[agent_id],
                        opponent_tensor,
                        deterministic=True,
                    )
                    states[agent_id] = self._detach_state(model_output.posterior_state)
                    actor_output = self.bundle.actors[agent_id].act(
                        model_output.posterior_state.features,
                        deterministic=False,
                    )
                actions[agent_id] = int(actor_output.action.item())

            step = self.env.step(actions)
            opponent_contexts = {
                agent_id: build_opponent_context(
                    agent_id,
                    self.env.agent_ids,
                    actions,
                    step.alive,
                    self.env.action_dim,
                )
                for agent_id in self.env.agent_ids
            }
            self._store_replay_step(observations, actions, step)
            for agent_id, reward in step.rewards.items():
                self.reward_totals[agent_id] += reward
            observations = step.observations
            infos = step.infos
            prev_actions = actions.copy()
            prev_opponent_contexts = opponent_contexts
            self.env_steps += 1

            if self._can_update_replay():
                metrics = self._run_updates()
                self.logger.log(
                    {
                        "phase": "train",
                        "algorithm": self.cfg.algorithm.name,
                        "env_steps": self.env_steps,
                        "episodes": self.episodes,
                        **metrics,
                    }
                )

            if step.done:
                self.episodes += 1
                self.episode_id += 1
                observations = self.env.reset(seed=self.cfg.seed + self.episode_id)
                infos = self.env.last_infos
                states, prev_actions, prev_opponent_contexts = self._initial_rollout_state()

            if self.env_steps >= next_eval_at:
                self.latest_eval_metrics = self.evaluate(self.cfg.training.eval_episodes)
                self.logger.log(
                    {
                        "phase": "eval",
                        "algorithm": self.cfg.algorithm.name,
                        "env_steps": self.env_steps,
                        **self.latest_eval_metrics,
                    }
                )
                observations = self.env.reset(seed=self.cfg.seed + self.episode_id + 1000)
                infos = self.env.last_infos
                states, prev_actions, prev_opponent_contexts = self._initial_rollout_state()
                next_eval_at += self.cfg.training.eval_interval_steps

            if self.env_steps >= next_save_at:
                self._save_checkpoint()
                next_save_at += self.cfg.training.save_interval_steps

        self._save_checkpoint()
        if not self.latest_eval_metrics:
            self.latest_eval_metrics = self.evaluate(self.cfg.training.eval_episodes)
        return TrainingSummary(
            algorithm=self.cfg.algorithm.name,
            env_mode=self.cfg.env.mode,
            learner_setup=self.cfg.algorithm.learner_setup,
            total_env_steps=self.env_steps,
            episodes=self.episodes,
            reward_totals=self.reward_totals,
            replay_size=len(self.replay),
            latest_checkpoint_path=str(self.checkpoint_path),
            latest_eval_metrics=self.latest_eval_metrics,
        )

    def evaluate(self, episodes: int, opponent_policy: str | None = None) -> dict[str, float]:
        eval_opponents = FixedOpponentManager(
            self.env,
            policy_name=opponent_policy or self.cfg.algorithm.opponent_policy,
            controlled_agent_ids=self.controlled_agent_ids,
            seed=self.cfg.seed + 777,
        )
        episode_rewards: list[float] = []
        wins = 0
        for episode in range(episodes):
            observations = self.env.reset(seed=self.cfg.seed + 20_000 + episode)
            infos = self.env.last_infos
            states, prev_actions, prev_opponent_contexts = self._initial_rollout_state()
            done = False
            reward_total = 0.0
            while not done:
                actions = eval_opponents.actions(observations, infos)
                for agent_id in self.controlled_agent_ids:
                    world_model = self.bundle.world_models[agent_id]
                    obs_tensor = self._obs_tensor(observations[agent_id][None])
                    prev_action = torch.as_tensor([prev_actions[agent_id]], device=self.device, dtype=torch.long)
                    opponent_context = prev_opponent_contexts[agent_id]
                    opponent_tensor = (
                        torch.as_tensor(opponent_context[None], device=self.device, dtype=torch.float32)
                        if world_model.opponent_action_dim
                        else None
                    )
                    with torch.no_grad():
                        model_output = world_model.observe(
                            obs_tensor,
                            prev_action,
                            states[agent_id],
                            opponent_tensor,
                            deterministic=True,
                        )
                        states[agent_id] = self._detach_state(model_output.posterior_state)
                        actor_output = self.bundle.actors[agent_id].act(
                            model_output.posterior_state.features,
                            deterministic=True,
                        )
                    actions[agent_id] = int(actor_output.action.item())
                step = self.env.step(actions)
                observations = step.observations
                infos = step.infos
                prev_actions = actions.copy()
                prev_opponent_contexts = {
                    agent_id: build_opponent_context(
                        agent_id,
                        self.env.agent_ids,
                        actions,
                        step.alive,
                        self.env.action_dim,
                    )
                    for agent_id in self.env.agent_ids
                }
                reward_total += float(sum(step.rewards[agent_id] for agent_id in self.controlled_agent_ids))
                done = step.done
                if done and any(step.raw_rewards[agent_id] > 0.0 for agent_id in self.controlled_agent_ids):
                    wins += 1
            episode_rewards.append(reward_total / len(self.controlled_agent_ids))
        return {
            "eval_episodes": float(episodes),
            "eval_mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "eval_win_rate": float(wins / max(episodes, 1)),
        }

    def _can_update_replay(self) -> bool:
        if len(self.replay) < self.cfg.algorithm.dreamer.warmup_steps:
            return False
        sequence_length = self.cfg.algorithm.dreamer.sequence_length
        return self.replay.num_valid_sequences(sequence_length) > 0

    def _run_updates(self) -> dict[str, float]:
        metrics = {
            "world_model_loss": 0.0,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
        }
        valid_sequences = self.replay.num_valid_sequences(self.cfg.algorithm.dreamer.sequence_length)
        if valid_sequences <= 0:
            return metrics
        for _ in range(self.cfg.algorithm.dreamer.updates_per_collect):
            batch = self.replay.sample_sequences(
                batch_size=min(self.cfg.algorithm.dreamer.batch_size, valid_sequences),
                sequence_length=self.cfg.algorithm.dreamer.sequence_length,
                agent_ids=self.env.agent_ids,
            )
            world_metrics, start_states = self._update_world_models(batch)
            actor_metrics = self._update_actors_and_critics(batch, start_states)
            for key, value in {**world_metrics, **actor_metrics}.items():
                metrics[key] = metrics.get(key, 0.0) + value
        scale = float(self.cfg.algorithm.dreamer.updates_per_collect)
        return {key: value / scale for key, value in metrics.items()}

    def _update_world_models(self, batch: Any) -> tuple[dict[str, float], dict[str, RSSMState]]:
        for optimizer in self.world_optimizers.values():
            optimizer.zero_grad()
        total_loss = torch.zeros((), device=self.device)
        metrics = {
            "world_model_loss": 0.0,
            "world_model_kl": 0.0,
            "world_model_reward": 0.0,
            "world_model_continuation": 0.0,
            "world_model_reconstruction": 0.0,
        }
        start_states: dict[str, RSSMState] = {}
        for agent_id in self.controlled_agent_ids:
            loss, agent_metrics, start_state = self._world_model_loss(
                self.bundle.world_models[agent_id],
                batch.agents[agent_id],
            )
            total_loss = total_loss + loss
            start_states[agent_id] = self._detach_state(start_state)
            for key, value in agent_metrics.items():
                metrics[key] = metrics.get(key, 0.0) + value
        total_loss.backward()
        for agent_id, model in self.bundle.unique_world_models().items():
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.algorithm.dreamer.max_grad_norm)
            self.world_optimizers[agent_id].step()
        scale = float(len(self.controlled_agent_ids))
        return {key: value / scale for key, value in metrics.items()}, start_states

    def _world_model_loss(self, world_model: Any, agent_batch: Any) -> tuple[Tensor, dict[str, float], RSSMState]:
        observations = self._obs_tensor(agent_batch.next_observations)
        actions = torch.as_tensor(agent_batch.actions, device=self.device, dtype=torch.long)
        rewards = torch.as_tensor(agent_batch.rewards, device=self.device, dtype=torch.float32)
        continues = torch.as_tensor(agent_batch.continues, device=self.device, dtype=torch.float32)
        opponent_actions = (
            torch.as_tensor(agent_batch.opponent_actions, device=self.device, dtype=torch.float32)
            if world_model.opponent_action_dim
            else None
        )
        batch_size, sequence_length = actions.shape
        state = world_model.initial_state(batch_size, self.device)
        metrics = {
            "world_model_loss": 0.0,
            "world_model_kl": 0.0,
            "world_model_reward": 0.0,
            "world_model_continuation": 0.0,
            "world_model_reconstruction": 0.0,
        }
        total_loss = torch.zeros((), device=self.device)
        for timestep in range(sequence_length):
            model_output = world_model.observe(
                observations[:, timestep],
                actions[:, timestep],
                state,
                opponent_actions[:, timestep] if opponent_actions is not None else None,
                deterministic=False,
            )
            board_target, scalar_target = extract_observation_targets(
                observations[:, timestep],
                world_model.board_value_count,
            )
            kl_loss = torch.clamp(
                kl_divergence(model_output.posterior_state, model_output.prior_state).mean(),
                min=self.cfg.algorithm.dreamer.free_nats,
            )
            reward_loss = F.mse_loss(model_output.reward_prediction, rewards[:, timestep])
            continuation_loss = F.binary_cross_entropy_with_logits(
                model_output.continuation_logit,
                continues[:, timestep],
            )
            board_loss = F.cross_entropy(model_output.board_logits, board_target)
            scalar_loss = F.mse_loss(model_output.scalar_prediction, scalar_target)
            reconstruction_loss = board_loss + scalar_loss
            step_loss = (
                self.cfg.algorithm.dreamer.kl_scale * kl_loss
                + self.cfg.algorithm.dreamer.reward_scale * reward_loss
                + self.cfg.algorithm.dreamer.continuation_scale * continuation_loss
                + self.cfg.algorithm.dreamer.reconstruction_scale * reconstruction_loss
            )
            total_loss = total_loss + step_loss
            metrics["world_model_loss"] += float(step_loss.item())
            metrics["world_model_kl"] += float(kl_loss.item())
            metrics["world_model_reward"] += float(reward_loss.item())
            metrics["world_model_continuation"] += float(continuation_loss.item())
            metrics["world_model_reconstruction"] += float(reconstruction_loss.item())
            state = model_output.posterior_state
        scale = float(sequence_length)
        return total_loss / scale, {key: value / scale for key, value in metrics.items()}, state

    def _update_actors_and_critics(self, batch: Any, start_states: dict[str, RSSMState]) -> dict[str, float]:
        metrics = {"actor_loss": 0.0, "critic_loss": 0.0}
        for agent_id in self.controlled_agent_ids:
            actor_loss, critic_loss = self._actor_critic_loss(
                agent_id,
                start_states[agent_id],
                batch.agents[agent_id],
            )
            actor_optimizer = self.actor_optimizers[agent_id]
            critic_optimizer = self.critic_optimizers[agent_id]

            actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(
                self.bundle.actors[agent_id].parameters(),
                self.cfg.algorithm.dreamer.max_grad_norm,
            )
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.bundle.critics[agent_id].parameters(),
                self.cfg.algorithm.dreamer.max_grad_norm,
            )
            critic_optimizer.step()

            metrics["actor_loss"] += float(actor_loss.item())
            metrics["critic_loss"] += float(critic_loss.item())
        scale = float(len(self.controlled_agent_ids))
        return {key: value / scale for key, value in metrics.items()}

    def _actor_critic_loss(self, agent_id: str, start_state: RSSMState, agent_batch: Any) -> tuple[Tensor, Tensor]:
        world_model = self.bundle.world_models[agent_id]
        actor = self.bundle.actors[agent_id]
        critic = self.bundle.critics[agent_id]
        state = self._detach_state(start_state)
        opponent_context = (
            torch.as_tensor(agent_batch.opponent_actions[:, -1], device=self.device, dtype=torch.float32)
            if world_model.opponent_action_dim
            else None
        )
        rewards: list[Tensor] = []
        continuations: list[Tensor] = []
        log_probs: list[Tensor] = []
        entropies: list[Tensor] = []
        values: list[Tensor] = []
        for _ in range(self.cfg.algorithm.dreamer.imagination_horizon):
            features = state.features
            actor_output = actor.act(features, deterministic=False)
            values.append(critic(features))
            log_probs.append(actor_output.log_prob)
            entropies.append(actor_output.entropy)
            imagination = world_model.imagine(
                state,
                actor_output.action,
                opponent_context,
                deterministic=False,
            )
            rewards.append(imagination.reward_prediction)
            continuations.append(torch.sigmoid(imagination.continuation_logit))
            state = self._detach_state(imagination.next_state)
        bootstrap = critic(state.features).detach()
        returns = self._lambda_returns(rewards, continuations, values, bootstrap)
        actor_loss = torch.zeros((), device=self.device)
        critic_loss = torch.zeros((), device=self.device)
        for index, value in enumerate(values):
            advantage = returns[index] - value
            actor_loss = actor_loss - (
                log_probs[index] * advantage.detach()
                + self.cfg.algorithm.dreamer.entropy_coef * entropies[index]
            ).mean()
            critic_loss = critic_loss + F.mse_loss(value, returns[index].detach())
        scale = float(max(len(values), 1))
        return actor_loss / scale, critic_loss / scale

    def _lambda_returns(
        self,
        rewards: list[Tensor],
        continuations: list[Tensor],
        values: list[Tensor],
        bootstrap: Tensor,
    ) -> list[Tensor]:
        lambda_return = self.cfg.algorithm.dreamer.lambda_return
        gamma = self.cfg.algorithm.dreamer.gamma
        returns: list[Tensor] = []
        running = bootstrap
        for index in reversed(range(len(rewards))):
            next_value = bootstrap if index == len(rewards) - 1 else values[index + 1].detach()
            running = rewards[index].detach() + gamma * continuations[index].detach() * (
                (1.0 - lambda_return) * next_value + lambda_return * running
            )
            returns.insert(0, running)
        return returns

    def _initial_rollout_state(self) -> tuple[dict[str, RSSMState], dict[str, int], dict[str, np.ndarray]]:
        states = {
            agent_id: self.bundle.world_models[agent_id].initial_state(1, self.device)
            for agent_id in self.controlled_agent_ids
        }
        prev_actions = {agent_id: 0 for agent_id in self.env.agent_ids}
        prev_opponent_contexts = {
            agent_id: np.zeros(
                self.bundle.world_models[agent_id].opponent_action_dim,
                dtype=np.float32,
            )
            if agent_id in self.controlled_agent_ids
            else np.zeros(0, dtype=np.float32)
            for agent_id in self.env.agent_ids
        }
        return states, prev_actions, prev_opponent_contexts

    def _detach_state(self, state: RSSMState) -> RSSMState:
        return RSSMState(
            deter=state.deter.detach(),
            stoch=state.stoch.detach(),
            mean=state.mean.detach(),
            std=state.std.detach(),
        )

    def _obs_tensor(self, obs: np.ndarray) -> Tensor:
        return torch.as_tensor(obs, device=self.device, dtype=torch.float32)

    def _store_replay_step(self, observations: dict[str, np.ndarray], actions: dict[str, int], step: Any) -> None:
        self.replay.add(
            ReplayStep(
                episode_id=self.episode_id,
                observations={agent_id: obs.copy() for agent_id, obs in observations.items()},
                actions=actions.copy(),
                opponent_actions={
                    agent_id: build_opponent_context(
                        agent_id,
                        self.env.agent_ids,
                        actions,
                        step.alive,
                        self.env.action_dim,
                    )
                    for agent_id in self.env.agent_ids
                },
                rewards=step.rewards.copy(),
                raw_rewards=step.raw_rewards.copy(),
                next_observations={
                    agent_id: obs.copy()
                    for agent_id, obs in step.observations.items()
                },
                terminated=step.terminated.copy(),
                truncated=step.truncated.copy(),
                alive=step.alive.copy(),
                infos={agent_id: dict(info) for agent_id, info in step.infos.items()},
                events={agent_id: dict(event) for agent_id, event in step.events.items()},
            )
        )

    def _save_checkpoint(self) -> None:
        payload = {
            "env_steps": self.env_steps,
            "episodes": self.episodes,
            "episode_id": self.episode_id,
            "reward_totals": self.reward_totals,
            "latest_eval_metrics": self.latest_eval_metrics,
            "bundle": {
                "world_models": {
                    agent_id: model.state_dict()
                    for agent_id, model in self.bundle.world_models.items()
                },
                "actors": {
                    agent_id: actor.state_dict()
                    for agent_id, actor in self.bundle.actors.items()
                },
                "critics": {
                    agent_id: critic.state_dict()
                    for agent_id, critic in self.bundle.critics.items()
                },
            },
            "optimizers": {
                "world": {
                    agent_id: optimizer.state_dict()
                    for agent_id, optimizer in self.world_optimizers.items()
                },
                "actors": {
                    agent_id: optimizer.state_dict()
                    for agent_id, optimizer in self.actor_optimizers.items()
                },
                "critics": {
                    agent_id: optimizer.state_dict()
                    for agent_id, optimizer in self.critic_optimizers.items()
                },
            },
            "config": asdict(self.cfg),
        }
        torch.save(payload, self.checkpoint_path)

    def _load_checkpoint(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device)
        for agent_id, state_dict in payload.get("bundle", {}).get("world_models", {}).items():
            if agent_id in self.bundle.world_models:
                self.bundle.world_models[agent_id].load_state_dict(state_dict)
        for agent_id, state_dict in payload.get("bundle", {}).get("actors", {}).items():
            if agent_id in self.bundle.actors:
                self.bundle.actors[agent_id].load_state_dict(state_dict)
        for agent_id, state_dict in payload.get("bundle", {}).get("critics", {}).items():
            if agent_id in self.bundle.critics:
                self.bundle.critics[agent_id].load_state_dict(state_dict)
        for agent_id, optimizer_state in payload.get("optimizers", {}).get("world", {}).items():
            if agent_id in self.world_optimizers:
                self.world_optimizers[agent_id].load_state_dict(optimizer_state)
        for agent_id, optimizer_state in payload.get("optimizers", {}).get("actors", {}).items():
            if agent_id in self.actor_optimizers:
                self.actor_optimizers[agent_id].load_state_dict(optimizer_state)
        for agent_id, optimizer_state in payload.get("optimizers", {}).get("critics", {}).items():
            if agent_id in self.critic_optimizers:
                self.critic_optimizers[agent_id].load_state_dict(optimizer_state)
        self.env_steps = int(payload.get("env_steps", 0))
        self.episodes = int(payload.get("episodes", 0))
        self.episode_id = int(payload.get("episode_id", 0))
        self.reward_totals.update(payload.get("reward_totals", {}))
        self.latest_eval_metrics = dict(payload.get("latest_eval_metrics", {}))
