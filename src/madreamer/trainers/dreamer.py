from __future__ import annotations

import copy
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
from madreamer.envs.pommerman import (
    BOMB_ACTION,
    BOMB_TILE,
    MOVE_DELTAS,
    PASSABLE_TILES,
    POMMERMAN_ACTION_DIM,
    pommerman_action_mask_from_encoded,
)
from madreamer.models.world_model import RSSMState, extract_observation_targets, kl_divergence
from madreamer.opponents import FixedOpponentManager
from madreamer.replay import MultiAgentReplayBuffer, ReplayStep, build_opponent_context
from madreamer.tracking import JsonlLogger
from madreamer.trainers.common import TrainingProgress, TrainingSummary, ensure_dir


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
        self.best_checkpoint_path = checkpoint_dir / f"{cfg.experiment_name}_{cfg.algorithm.name}_best.pt"
        self.reward_totals = {agent_id: 0.0 for agent_id in self.env.agent_ids}
        self.env_steps = 0
        self.episodes = 0
        self.episode_id = 0
        self.latest_eval_metrics: dict[str, float] = {}
        self.best_eval_metrics: dict[str, float] = {}
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
        self.target_critics = {
            agent_id: copy.deepcopy(self.bundle.critics[agent_id]).eval()
            for agent_id in self.controlled_agent_ids
        }
        for target_critic in self.target_critics.values():
            for parameter in target_critic.parameters():
                parameter.requires_grad_(False)
        if cfg.training.resume_checkpoint:
            self._load_checkpoint(Path(cfg.training.resume_checkpoint))

    def run(self) -> TrainingSummary:
        observations = self.env.reset(seed=self.cfg.seed)
        infos = self.env.last_infos
        states, prev_actions, prev_opponent_contexts = self._initial_rollout_state()
        next_eval_at = self.cfg.training.eval_interval_steps
        next_save_at = self.cfg.training.save_interval_steps
        progress = TrainingProgress(
            total_steps=self.cfg.training.total_steps,
            label=f"{self.cfg.experiment_name}/{self.cfg.algorithm.name}",
        )
        progress.update(
            self.env_steps,
            episodes=self.episodes,
            latest_eval_metrics=self.latest_eval_metrics,
            force=True,
        )

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
                        action_mask=self._real_action_mask_tensor(observations, infos, agent_id),
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
            progress.update(
                self.env_steps,
                episodes=self.episodes,
                latest_eval_metrics=self.latest_eval_metrics,
            )

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
                progress.update(
                    self.env_steps,
                    episodes=self.episodes,
                    latest_eval_metrics=self.latest_eval_metrics,
                    phase="eval",
                    force=True,
                )
                self.latest_eval_metrics = self.evaluate(self.cfg.training.eval_episodes)
                self._maybe_save_best_checkpoint()
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
                progress.update(
                    self.env_steps,
                    episodes=self.episodes,
                    latest_eval_metrics=self.latest_eval_metrics,
                    force=True,
                )

            if self.env_steps >= next_save_at:
                self._save_checkpoint()
                next_save_at += self.cfg.training.save_interval_steps

        self._save_checkpoint()
        if not self.latest_eval_metrics:
            progress.update(
                self.env_steps,
                episodes=self.episodes,
                latest_eval_metrics=self.latest_eval_metrics,
                phase="eval",
                force=True,
            )
            self.latest_eval_metrics = self.evaluate(self.cfg.training.eval_episodes)
            self._maybe_save_best_checkpoint()
        progress.finish(
            self.env_steps,
            episodes=self.episodes,
            latest_eval_metrics=self.latest_eval_metrics,
        )
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
            best_checkpoint_path=str(self.best_checkpoint_path) if self.best_checkpoint_path.exists() else None,
            best_eval_metrics=self.best_eval_metrics,
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
                            action_mask=self._real_action_mask_tensor(observations, infos, agent_id),
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
        train_every_steps = max(1, self.cfg.algorithm.dreamer.train_every_steps)
        if self.env_steps % train_every_steps != 0:
            return False
        sequence_length = self.cfg.algorithm.dreamer.sequence_length
        return self.replay.num_valid_sequences(sequence_length) > 0

    def _run_updates(self) -> dict[str, float]:
        metrics = {
            "world_model_loss": 0.0,
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "policy_updates_enabled": float(self._policy_updates_enabled()),
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
            actor_metrics = (
                self._update_actors_and_critics(batch, start_states)
                if self._policy_updates_enabled()
                else {"actor_loss": 0.0, "critic_loss": 0.0, "imagined_action_entropy": 0.0}
            )
            behavior_metrics = self._batch_behavior_metrics(batch)
            for key, value in {**world_metrics, **actor_metrics, **behavior_metrics}.items():
                metrics[key] = metrics.get(key, 0.0) + value
        scale = float(self.cfg.algorithm.dreamer.updates_per_collect)
        averaged = {key: value / scale for key, value in metrics.items()}
        averaged["policy_updates_enabled"] = float(self._policy_updates_enabled())
        averaged["policy_warmup_remaining"] = float(
            max(0, self.cfg.algorithm.dreamer.policy_warmup_steps - self.env_steps)
        )
        return averaged

    def _policy_updates_enabled(self) -> bool:
        return self.env_steps >= self.cfg.algorithm.dreamer.policy_warmup_steps

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
            "world_model_board_accuracy": 0.0,
            "world_model_kl_raw": 0.0,
            "world_model_reward_mae": 0.0,
            "world_model_continuation_accuracy": 0.0,
            "world_model_scalar_mae": 0.0,
            "world_model_feature_mae": 0.0,
            "world_model_bomb_blast_mae": 0.0,
            "world_model_bomb_life_mae": 0.0,
            "world_model_position_mae": 0.0,
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
        observations = self._obs_tensor(agent_batch.observations)
        next_observations = self._obs_tensor(agent_batch.next_observations)
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
            "world_model_board_accuracy": 0.0,
            "world_model_kl_raw": 0.0,
            "world_model_reward_mae": 0.0,
            "world_model_continuation_accuracy": 0.0,
            "world_model_scalar_mae": 0.0,
            "world_model_feature_mae": 0.0,
            "world_model_bomb_blast_mae": 0.0,
            "world_model_bomb_life_mae": 0.0,
            "world_model_position_mae": 0.0,
        }
        zero_prev_actions = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        zero_opponent_actions = (
            torch.zeros(batch_size, world_model.opponent_action_dim, device=self.device, dtype=torch.float32)
            if world_model.opponent_action_dim
            else None
        )
        state = world_model.observe(
            observations[:, 0],
            zero_prev_actions,
            state,
            zero_opponent_actions,
            deterministic=False,
        ).posterior_state
        posterior_states: list[RSSMState] = []
        total_loss = torch.zeros((), device=self.device)
        for timestep in range(sequence_length):
            model_output = world_model.observe(
                next_observations[:, timestep],
                actions[:, timestep],
                state,
                opponent_actions[:, timestep] if opponent_actions is not None else None,
                deterministic=False,
            )
            board_target, scalar_target = extract_observation_targets(
                next_observations[:, timestep],
                world_model.board_value_count,
            )
            kl_values = kl_divergence(model_output.posterior_state, model_output.prior_state)
            kl_loss = self._apply_free_nats(kl_values, self.cfg.algorithm.dreamer.free_nats)
            reward_loss = F.mse_loss(model_output.reward_prediction, rewards[:, timestep])
            continuation_loss = F.binary_cross_entropy_with_logits(
                model_output.continuation_logit,
                continues[:, timestep],
            )
            board_loss = F.cross_entropy(
                model_output.board_logits,
                board_target,
                weight=self._board_loss_weights(board_target, world_model.board_value_count),
            )
            scalar_loss = F.mse_loss(model_output.scalar_prediction, scalar_target)
            reconstruction_loss = board_loss + scalar_loss
            board_prediction = model_output.board_logits.argmax(dim=1)
            board_accuracy = (board_prediction == board_target).float().mean()
            reward_mae = (model_output.reward_prediction - rewards[:, timestep]).abs().mean()
            continuation_prediction = (torch.sigmoid(model_output.continuation_logit) >= 0.5).float()
            continuation_accuracy = (continuation_prediction == continues[:, timestep]).float().mean()
            scalar_mae = (model_output.scalar_prediction - scalar_target).abs().mean()
            bomb_blast_mae = (model_output.scalar_prediction[:, 0] - scalar_target[:, 0]).abs().mean()
            bomb_life_mae = (model_output.scalar_prediction[:, 1] - scalar_target[:, 1]).abs().mean()
            position_mae = (model_output.scalar_prediction[:, 2] - scalar_target[:, 2]).abs().mean()
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
            metrics["world_model_board_accuracy"] += float(board_accuracy.item())
            metrics["world_model_kl_raw"] += float(kl_values.mean().item())
            metrics["world_model_reward_mae"] += float(reward_mae.item())
            metrics["world_model_continuation_accuracy"] += float(continuation_accuracy.item())
            metrics["world_model_scalar_mae"] += float(scalar_mae.item())
            metrics["world_model_feature_mae"] += float(scalar_mae.item())
            metrics["world_model_bomb_blast_mae"] += float(bomb_blast_mae.item())
            metrics["world_model_bomb_life_mae"] += float(bomb_life_mae.item())
            metrics["world_model_position_mae"] += float(position_mae.item())
            state = model_output.posterior_state
            posterior_states.append(self._detach_state(state))
        scale = float(sequence_length)
        return total_loss / scale, {key: value / scale for key, value in metrics.items()}, self._concat_states(posterior_states)

    def _board_loss_weights(self, board_target: Tensor, class_count: int) -> Tensor | None:
        balance = self.cfg.algorithm.dreamer.board_class_balance
        if balance <= 0.0:
            return None
        balance = min(balance, 1.0)
        counts = torch.bincount(board_target.reshape(-1), minlength=class_count).float()
        present = counts > 0
        if int(present.sum().item()) <= 1:
            return None
        weights = torch.ones(class_count, device=board_target.device)
        inverse_frequency = torch.zeros_like(weights)
        inverse_frequency[present] = torch.sqrt(counts[present].sum() / counts[present].clamp_min(1.0))
        inverse_frequency[present] = inverse_frequency[present] / inverse_frequency[present].mean().clamp_min(1e-6)
        weights[present] = (1.0 - balance) + balance * inverse_frequency[present]
        return weights

    def _update_actors_and_critics(self, batch: Any, start_states: dict[str, RSSMState]) -> dict[str, float]:
        metrics = {"actor_loss": 0.0, "critic_loss": 0.0, "imagined_action_entropy": 0.0}
        for agent_id in self.controlled_agent_ids:
            actor_loss, critic_loss, agent_metrics = self._actor_critic_loss(
                agent_id,
                start_states[agent_id],
                batch.agents[agent_id],
            )
            actor_optimizer = self.actor_optimizers[agent_id]
            critic_optimizer = self.critic_optimizers[agent_id]

            actor_optimizer.zero_grad()
            actor_loss.backward()
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
            self._update_target_critic(agent_id)

            metrics["actor_loss"] += float(actor_loss.item())
            metrics["critic_loss"] += float(critic_loss.item())
            for key, value in agent_metrics.items():
                metrics[key] = metrics.get(key, 0.0) + value
        scale = float(len(self.controlled_agent_ids))
        return {key: value / scale for key, value in metrics.items()}

    def _actor_critic_loss(self, agent_id: str, start_state: RSSMState, agent_batch: Any) -> tuple[Tensor, Tensor, dict[str, float]]:
        world_model = self.bundle.world_models[agent_id]
        actor = self.bundle.actors[agent_id]
        critic = self.bundle.critics[agent_id]
        target_critic = self.target_critics[agent_id]
        state = self._detach_state(start_state)
        opponent_context = self._opponent_context_for_imagination(world_model, agent_batch, state)
        rewards: list[Tensor] = []
        continuations: list[Tensor] = []
        log_probs: list[Tensor] = []
        entropies: list[Tensor] = []
        values: list[Tensor] = []
        target_values: list[Tensor] = []
        imagined_actions: list[Tensor] = []
        for _ in range(self.cfg.algorithm.dreamer.imagination_horizon):
            features = state.features
            action_mask = self._imagined_action_mask(world_model, state)
            actor_output = actor.act(features, deterministic=False, action_mask=action_mask)
            values.append(critic(features))
            with torch.no_grad():
                target_values.append(target_critic(features))
            log_probs.append(actor_output.log_prob)
            entropies.append(actor_output.entropy)
            imagined_actions.append(actor_output.action)
            imagination = world_model.imagine(
                state,
                actor_output.action,
                opponent_context,
                deterministic=False,
            )
            rewards.append(imagination.reward_prediction)
            continuations.append(torch.sigmoid(imagination.continuation_logit))
            state = self._detach_state(imagination.next_state)
        with torch.no_grad():
            bootstrap = target_critic(state.features)
        returns = self._lambda_returns(rewards, continuations, target_values, bootstrap)
        actor_loss = torch.zeros((), device=self.device)
        critic_loss = torch.zeros((), device=self.device)
        advantages = [returns[index] - value for index, value in enumerate(values)]
        normalized_advantages = self._normalize_advantages(advantages)
        for index, value in enumerate(values):
            actor_loss = actor_loss - (
                log_probs[index] * normalized_advantages[index].detach()
                + self.cfg.algorithm.dreamer.entropy_coef * entropies[index]
            ).mean()
            critic_loss = critic_loss + F.mse_loss(value, returns[index].detach())
        scale = float(max(len(values), 1))
        metrics = {
            "imagined_action_entropy": float(torch.stack(entropies).mean().item()) if entropies else 0.0,
        }
        if imagined_actions:
            action_tensor = torch.stack(imagined_actions).reshape(-1)
            for action in range(self.env.action_dim):
                metrics[f"imagined_action_{action}_rate"] = float((action_tensor == action).float().mean().item())
        return actor_loss / scale, critic_loss / scale, metrics

    def _real_action_mask_tensor(
        self,
        observations: dict[str, np.ndarray],
        infos: dict[str, dict[str, object]],
        agent_id: str,
    ) -> Tensor | None:
        if self.cfg.env.name != "pommerman" or self.env.action_dim != POMMERMAN_ACTION_DIM:
            return None
        info = infos.get(agent_id, {})
        mask = info.get("action_mask")
        if mask is None:
            mask = pommerman_action_mask_from_encoded(
                observations[agent_id],
                board_value_count=self.cfg.env.board_value_count,
                action_dim=self.env.action_dim,
            )
        mask_array = np.asarray(mask, dtype=np.float32)
        if mask_array.shape != (self.env.action_dim,):
            return None
        return torch.as_tensor(mask_array[None], device=self.device, dtype=torch.bool)

    def _imagined_action_mask(self, world_model: Any, state: RSSMState) -> Tensor | None:
        if (
            self.cfg.env.name != "pommerman"
            or self.env.action_dim != POMMERMAN_ACTION_DIM
            or world_model.obs_shape[0] < world_model.board_value_count + 6
        ):
            return None
        with torch.no_grad():
            _, _, board_logits, feature_prediction = world_model.decode(state)
            board = board_logits.argmax(dim=1)
            bomb_life = feature_prediction[:, 1] * 10.0
            batch_size, height, width = board.shape
            positions = feature_prediction[:, 2].reshape(batch_size, -1).argmax(dim=1)
            rows = torch.div(positions, width, rounding_mode="floor")
            cols = positions.remainder(width)
            mask = torch.zeros(batch_size, self.env.action_dim, device=self.device, dtype=torch.bool)
            mask[:, 0] = True
            can_kick = feature_prediction[:, 5].reshape(batch_size, -1).mean(dim=1) >= 0.5
            indices = torch.arange(batch_size, device=self.device)
            for action, (delta_row, delta_col) in MOVE_DELTAS.items():
                if action >= self.env.action_dim:
                    continue
                target_rows = rows + delta_row
                target_cols = cols + delta_col
                in_bounds = (
                    (target_rows >= 0)
                    & (target_rows < height)
                    & (target_cols >= 0)
                    & (target_cols < width)
                )
                clamped_rows = target_rows.clamp(0, height - 1)
                clamped_cols = target_cols.clamp(0, width - 1)
                target_tiles = board[indices, clamped_rows, clamped_cols]
                target_has_bomb = (target_tiles == BOMB_TILE) | (bomb_life[indices, clamped_rows, clamped_cols] > 0.05)
                target_passable = self._tile_in(target_tiles, PASSABLE_TILES)
                kick_rows = target_rows + delta_row
                kick_cols = target_cols + delta_col
                kick_in_bounds = (
                    (kick_rows >= 0)
                    & (kick_rows < height)
                    & (kick_cols >= 0)
                    & (kick_cols < width)
                )
                clamped_kick_rows = kick_rows.clamp(0, height - 1)
                clamped_kick_cols = kick_cols.clamp(0, width - 1)
                kick_tiles = board[indices, clamped_kick_rows, clamped_kick_cols]
                kick_clear = (
                    kick_in_bounds
                    & self._tile_in(kick_tiles, PASSABLE_TILES)
                    & (bomb_life[indices, clamped_kick_rows, clamped_kick_cols] <= 0.05)
                )
                mask[:, action] = in_bounds & (
                    (~target_has_bomb & target_passable)
                    | (target_has_bomb & can_kick & kick_clear)
                )
            if BOMB_ACTION < self.env.action_dim:
                ammo = feature_prediction[:, 3].reshape(batch_size, -1).mean(dim=1) * 10.0
                current_bomb = bomb_life[indices, rows, cols] > 0.05
                mask[:, BOMB_ACTION] = (ammo > 0.05) & ~current_bomb
            return mask

    @staticmethod
    def _tile_in(tiles: Tensor, values: set[int]) -> Tensor:
        result = torch.zeros_like(tiles, dtype=torch.bool)
        for value in values:
            result = result | (tiles == value)
        return result

    @staticmethod
    def _apply_free_nats(kl_values: Tensor, free_nats: float) -> Tensor:
        return torch.clamp(kl_values, min=free_nats).mean()

    def _opponent_context_for_imagination(
        self,
        world_model: Any,
        agent_batch: Any,
        state: RSSMState,
    ) -> Tensor | None:
        if not world_model.opponent_action_dim:
            return None
        opponent_actions = torch.as_tensor(agent_batch.opponent_actions, device=self.device, dtype=torch.float32)
        flat_opponent_actions = opponent_actions.reshape(-1, world_model.opponent_action_dim)
        if flat_opponent_actions.shape[0] == state.deter.shape[0]:
            return flat_opponent_actions
        return opponent_actions[:, -1]

    def _update_target_critic(self, agent_id: str) -> None:
        tau = self.cfg.algorithm.dreamer.critic_target_tau
        source = self.bundle.critics[agent_id]
        target = self.target_critics[agent_id]
        with torch.no_grad():
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.data.lerp_(source_param.data, tau)

    @staticmethod
    def _normalize_advantages(advantages: list[Tensor]) -> list[Tensor]:
        if not advantages:
            return []
        flat = torch.cat([advantage.reshape(-1) for advantage in advantages])
        mean = flat.mean()
        std = flat.std(unbiased=False)
        scale = std.clamp_min(1e-6)
        return [(advantage - mean) / scale for advantage in advantages]

    def _batch_behavior_metrics(self, batch: Any) -> dict[str, float]:
        actions = []
        safe_stop_rates = []
        blocked_move_rates = []
        useful_bomb_rates = []
        wasted_bomb_rates = []
        tie_rates = []
        for agent_id in self.controlled_agent_ids:
            agent_batch = batch.agents[agent_id]
            actions.append(torch.as_tensor(agent_batch.actions, device=self.device, dtype=torch.long).reshape(-1))
            if "safe_stop" in agent_batch.events:
                safe_stop_rates.append(float(agent_batch.events["safe_stop"].mean()))
            if "blocked_move" in agent_batch.events:
                blocked_move_rates.append(float(agent_batch.events["blocked_move"].mean()))
            if "useful_bomb" in agent_batch.events:
                useful_bomb_rates.append(float(agent_batch.events["useful_bomb"].mean()))
            if "wasted_bomb" in agent_batch.events:
                wasted_bomb_rates.append(float(agent_batch.events["wasted_bomb"].mean()))
            if "tied" in agent_batch.events:
                tie_rates.append(float(agent_batch.events["tied"].mean()))
        if actions:
            action_tensor = torch.cat(actions)
        else:
            action_tensor = torch.zeros(0, device=self.device, dtype=torch.long)
        metrics = {
            "behavior_safe_stop_rate": float(np.mean(safe_stop_rates)) if safe_stop_rates else 0.0,
            "behavior_blocked_move_rate": float(np.mean(blocked_move_rates)) if blocked_move_rates else 0.0,
            "behavior_useful_bomb_rate": float(np.mean(useful_bomb_rates)) if useful_bomb_rates else 0.0,
            "behavior_wasted_bomb_rate": float(np.mean(wasted_bomb_rates)) if wasted_bomb_rates else 0.0,
            "behavior_tie_rate": float(np.mean(tie_rates)) if tie_rates else 0.0,
        }
        for action in range(self.env.action_dim):
            metrics[f"behavior_action_{action}_rate"] = (
                float((action_tensor == action).float().mean().item())
                if action_tensor.numel()
                else 0.0
            )
        return metrics

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

    def _concat_states(self, states: list[RSSMState]) -> RSSMState:
        if not states:
            raise ValueError("Cannot build imagination starts from an empty sequence.")
        return RSSMState(
            deter=torch.cat([state.deter for state in states], dim=0),
            stoch=torch.cat([state.stoch for state in states], dim=0),
            mean=torch.cat([state.mean for state in states], dim=0),
            std=torch.cat([state.std for state in states], dim=0),
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

    def _save_checkpoint(self, path: Path | None = None) -> None:
        destination = path or self.checkpoint_path
        payload = {
            "env_steps": self.env_steps,
            "episodes": self.episodes,
            "episode_id": self.episode_id,
            "reward_totals": self.reward_totals,
            "latest_eval_metrics": self.latest_eval_metrics,
            "best_eval_metrics": self.best_eval_metrics,
            "best_checkpoint_path": str(self.best_checkpoint_path),
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
                "target_critics": {
                    agent_id: critic.state_dict()
                    for agent_id, critic in self.target_critics.items()
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
        torch.save(payload, destination)

    def _maybe_save_best_checkpoint(self) -> None:
        if not self.latest_eval_metrics:
            return
        if self._is_better_eval(self.latest_eval_metrics, self.best_eval_metrics):
            self.best_eval_metrics = dict(self.latest_eval_metrics)
            self._save_checkpoint(self.best_checkpoint_path)

    def _is_better_eval(self, candidate: dict[str, float], incumbent: dict[str, float]) -> bool:
        if not candidate:
            return False
        if not incumbent:
            return True
        return self._eval_score(candidate) > self._eval_score(incumbent)

    def _eval_score(self, metrics: dict[str, float]) -> tuple[float, float]:
        return (
            self._finite_metric(metrics, "eval_win_rate"),
            self._finite_metric(metrics, "eval_mean_reward"),
        )

    @staticmethod
    def _finite_metric(metrics: dict[str, float], key: str) -> float:
        value = float(metrics.get(key, float("-inf")))
        return value if np.isfinite(value) else float("-inf")

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
        loaded_target_critics: set[str] = set()
        for agent_id, state_dict in payload.get("bundle", {}).get("target_critics", {}).items():
            if agent_id in self.target_critics:
                self.target_critics[agent_id].load_state_dict(state_dict)
                loaded_target_critics.add(agent_id)
        for agent_id, target_critic in self.target_critics.items():
            if agent_id not in loaded_target_critics and agent_id in self.bundle.critics:
                target_critic.load_state_dict(self.bundle.critics[agent_id].state_dict())
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
        self.best_eval_metrics = dict(payload.get("best_eval_metrics", {}))
