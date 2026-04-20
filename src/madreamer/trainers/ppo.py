from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from madreamer.builders import ModuleBundle
from madreamer.config import ExperimentConfig
from madreamer.envs.base import MultiAgentEnv
from madreamer.opponents import FixedOpponentManager
from madreamer.replay import MultiAgentReplayBuffer, ReplayStep, build_opponent_context
from madreamer.tracking import JsonlLogger
from madreamer.trainers.common import TrainingSummary, ensure_dir


@dataclass
class AgentRollout:
    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[float] = field(default_factory=list)


class PPOCollector:
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
        self.optimizers = {
            agent_id: torch.optim.Adam(
                self.bundle.ppo_policies[agent_id].parameters(),
                lr=cfg.algorithm.ppo.learning_rate,
            )
            for agent_id in self.controlled_agent_ids
        }
        self.logger = JsonlLogger(cfg.training.log_dir)
        checkpoint_dir = ensure_dir(cfg.training.checkpoint_dir)
        self.checkpoint_path = checkpoint_dir / f"{cfg.experiment_name}_ppo_latest.pt"
        self.env_steps = 0
        self.episodes = 0
        self.episode_id = 0
        self.reward_totals = {agent_id: 0.0 for agent_id in self.env.agent_ids}
        self.latest_eval_metrics: dict[str, float] = {}
        if cfg.training.resume_checkpoint:
            self._load_checkpoint(Path(cfg.training.resume_checkpoint))

    def run(self) -> TrainingSummary:
        observations = self.env.reset(seed=self.cfg.seed)
        infos = self.env.last_infos
        next_eval_at = self.cfg.training.eval_interval_steps
        next_save_at = self.cfg.training.save_interval_steps

        while self.env_steps < self.cfg.training.total_steps:
            rollouts = {agent_id: AgentRollout() for agent_id in self.controlled_agent_ids}
            collected = 0
            while (
                collected < self.cfg.algorithm.ppo.rollout_steps
                and self.env_steps < self.cfg.training.total_steps
            ):
                actions = self.opponents.actions(observations, infos)
                for agent_id in self.controlled_agent_ids:
                    obs_tensor = self._obs_tensor(observations[agent_id][None])
                    with torch.no_grad():
                        output = self.bundle.ppo_policies[agent_id].act(obs_tensor, deterministic=False)
                    actions[agent_id] = int(output.action.item())
                    rollout = rollouts[agent_id]
                    rollout.observations.append(observations[agent_id].copy())
                    rollout.actions.append(actions[agent_id])
                    rollout.log_probs.append(float(output.log_prob.item()))
                    rollout.values.append(float(output.value.item()))

                step = self.env.step(actions)
                self._store_replay_step(observations, actions, step)
                for agent_id in self.controlled_agent_ids:
                    rollouts[agent_id].rewards.append(float(step.rewards[agent_id]))
                    rollouts[agent_id].dones.append(float(step.terminated[agent_id] or step.truncated[agent_id]))
                for agent_id, reward in step.rewards.items():
                    self.reward_totals[agent_id] += reward
                observations = step.observations
                infos = step.infos
                self.env_steps += 1
                collected += 1

                if step.done:
                    self.episodes += 1
                    self.episode_id += 1
                    observations = self.env.reset(seed=self.cfg.seed + self.episode_id)
                    infos = self.env.last_infos

            for agent_id in self.controlled_agent_ids:
                next_value = 0.0
                if rollouts[agent_id].observations and not rollouts[agent_id].dones[-1]:
                    with torch.no_grad():
                        obs_tensor = self._obs_tensor(observations[agent_id][None])
                        next_output = self.bundle.ppo_policies[agent_id].act(obs_tensor, deterministic=True)
                    next_value = float(next_output.value.item())
                self._update_agent(agent_id, rollouts[agent_id], next_value)

            metrics = {
                "phase": "train",
                "algorithm": "ppo",
                "env_steps": self.env_steps,
                "episodes": self.episodes,
                "controlled_agents": list(self.controlled_agent_ids),
                "reward_totals": self.reward_totals,
            }
            self.logger.log(metrics)

            if self.env_steps >= next_eval_at:
                self.latest_eval_metrics = self.evaluate(self.cfg.training.eval_episodes)
                self.logger.log(
                    {
                        "phase": "eval",
                        "algorithm": "ppo",
                        "env_steps": self.env_steps,
                        **self.latest_eval_metrics,
                    }
                )
                observations = self.env.reset(seed=self.cfg.seed + self.episode_id + 1000)
                infos = self.env.last_infos
                next_eval_at += self.cfg.training.eval_interval_steps

            if self.env_steps >= next_save_at:
                self._save_checkpoint()
                next_save_at += self.cfg.training.save_interval_steps

        self._save_checkpoint()
        if not self.latest_eval_metrics:
            self.latest_eval_metrics = self.evaluate(self.cfg.training.eval_episodes)
        return TrainingSummary(
            algorithm="ppo",
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
        rewards: list[float] = []
        wins = 0
        eval_opponents = FixedOpponentManager(
            self.env,
            policy_name=opponent_policy or self.cfg.algorithm.opponent_policy,
            controlled_agent_ids=self.controlled_agent_ids,
            seed=self.cfg.seed + 999,
        )
        for episode in range(episodes):
            observations = self.env.reset(seed=self.cfg.seed + 10_000 + episode)
            infos = self.env.last_infos
            episode_reward = 0.0
            done = False
            while not done:
                actions = eval_opponents.actions(observations, infos)
                for agent_id in self.controlled_agent_ids:
                    obs_tensor = self._obs_tensor(observations[agent_id][None])
                    with torch.no_grad():
                        output = self.bundle.ppo_policies[agent_id].act(obs_tensor, deterministic=True)
                    actions[agent_id] = int(output.action.item())
                step = self.env.step(actions)
                observations = step.observations
                infos = step.infos
                episode_reward += float(sum(step.rewards[agent_id] for agent_id in self.controlled_agent_ids))
                done = step.done
                if done and any(step.raw_rewards[agent_id] > 0.0 for agent_id in self.controlled_agent_ids):
                    wins += 1
            rewards.append(episode_reward / len(self.controlled_agent_ids))
        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        return {
            "eval_episodes": float(episodes),
            "eval_mean_reward": mean_reward,
            "eval_win_rate": float(wins / max(episodes, 1)),
        }

    def _update_agent(self, agent_id: str, rollout: AgentRollout, next_value: float) -> None:
        if not rollout.observations:
            return
        advantages, returns = self._compute_gae(rollout, next_value)
        observations = self._obs_tensor(np.asarray(rollout.observations))
        actions = torch.as_tensor(rollout.actions, device=self.device, dtype=torch.long)
        old_log_probs = torch.as_tensor(rollout.log_probs, device=self.device, dtype=torch.float32)
        returns_tensor = torch.as_tensor(returns, device=self.device, dtype=torch.float32)
        advantages_tensor = torch.as_tensor(advantages, device=self.device, dtype=torch.float32)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-6)

        network = self.bundle.ppo_policies[agent_id]
        optimizer = self.optimizers[agent_id]
        total_samples = actions.shape[0]
        minibatch_size = min(self.cfg.algorithm.ppo.minibatch_size, total_samples)
        for _ in range(self.cfg.algorithm.ppo.update_epochs):
            permutation = torch.randperm(total_samples, device=self.device)
            for start in range(0, total_samples, minibatch_size):
                indices = permutation[start : start + minibatch_size]
                log_prob, entropy, value = network.evaluate_actions(observations[indices], actions[indices])
                ratio = torch.exp(log_prob - old_log_probs[indices])
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.cfg.algorithm.ppo.clip_ratio,
                    1.0 + self.cfg.algorithm.ppo.clip_ratio,
                )
                policy_loss = -torch.min(
                    ratio * advantages_tensor[indices],
                    clipped_ratio * advantages_tensor[indices],
                ).mean()
                value_loss = F.mse_loss(value, returns_tensor[indices])
                loss = (
                    policy_loss
                    + self.cfg.algorithm.ppo.value_coef * value_loss
                    - self.cfg.algorithm.ppo.entropy_coef * entropy.mean()
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    network.parameters(),
                    self.cfg.algorithm.ppo.max_grad_norm,
                )
                optimizer.step()

    def _compute_gae(self, rollout: AgentRollout, next_value: float) -> tuple[np.ndarray, np.ndarray]:
        rewards = np.asarray(rollout.rewards, dtype=np.float32)
        values = np.asarray(rollout.values + [next_value], dtype=np.float32)
        dones = np.asarray(rollout.dones, dtype=np.float32)
        advantages = np.zeros_like(rewards)
        last_advantage = 0.0
        gamma = self.cfg.algorithm.ppo.gamma
        gae_lambda = self.cfg.algorithm.ppo.gae_lambda
        for index in reversed(range(len(rewards))):
            mask = 1.0 - dones[index]
            delta = rewards[index] + gamma * values[index + 1] * mask - values[index]
            last_advantage = delta + gamma * gae_lambda * mask * last_advantage
            advantages[index] = last_advantage
        returns = advantages + values[:-1]
        return advantages, returns

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
                "ppo_policies": {
                    agent_id: network.state_dict()
                    for agent_id, network in self.bundle.ppo_policies.items()
                }
            },
            "optimizers": {
                agent_id: optimizer.state_dict()
                for agent_id, optimizer in self.optimizers.items()
            },
            "config": asdict(self.cfg),
        }
        torch.save(payload, self.checkpoint_path)

    def _load_checkpoint(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device)
        for agent_id, state_dict in payload.get("bundle", {}).get("ppo_policies", {}).items():
            if agent_id in self.bundle.ppo_policies:
                self.bundle.ppo_policies[agent_id].load_state_dict(state_dict)
        for agent_id, optimizer_state in payload.get("optimizers", {}).items():
            if agent_id in self.optimizers:
                self.optimizers[agent_id].load_state_dict(optimizer_state)
        self.env_steps = int(payload.get("env_steps", 0))
        self.episodes = int(payload.get("episodes", 0))
        self.episode_id = int(payload.get("episode_id", 0))
        self.reward_totals.update(payload.get("reward_totals", {}))
        self.latest_eval_metrics = dict(payload.get("latest_eval_metrics", {}))
