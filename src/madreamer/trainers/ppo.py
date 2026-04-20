from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
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


class PPOTrainer:
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
        self.optimizers = {
            agent_id: torch.optim.Adam(
                self.bundle.ppo_policies[agent_id].parameters(),
                lr=self.cfg.training.learning_rate,
            )
            for agent_id in self.trainable_agent_ids
        }
        self.output_dir = ensure_dir(Path(cfg.training.output_dir) / cfg.experiment_name)
        self.metrics_history: list[dict[str, Any]] = []

    def train(self) -> TrainerSummary:
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        observations = self.env.reset(seed=self.cfg.seed)
        infos = {agent_id: {} for agent_id in self.env.agent_ids}
        global_step = 0
        episodes = 0
        reward_totals = {agent_id: 0.0 for agent_id in self.env.agent_ids}
        episode_returns = {agent_id: 0.0 for agent_id in self.env.agent_ids}
        best_eval = float("-inf")
        checkpoint_path = self.output_dir / "latest.pt"

        while global_step < self.cfg.training.total_steps:
            rollout, observations, infos, step_count, episodes, reward_totals, episode_returns = self._collect_rollout(
                observations,
                infos,
                global_step,
                episodes,
                reward_totals,
                episode_returns,
            )
            global_step += step_count
            if step_count == 0:
                break
            metrics = self._update(rollout)
            eval_metrics = self.evaluate(self.cfg.training.eval_episodes)
            log_record = {
                "step": global_step,
                "episodes": episodes,
                "policy_loss": metrics["policy_loss"],
                "value_loss": metrics["value_loss"],
                "entropy": metrics["entropy"],
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
            observations = self.env.reset(seed=self.cfg.seed + 10_000 + offset)
            infos = {agent_id: {} for agent_id in self.env.agent_ids}
            done = False
            reward_total = 0.0
            while not done:
                actions: dict[str, int] = {}
                with torch.no_grad():
                    for agent_id in self.env.agent_ids:
                        if agent_id in self.trainable_agent_ids:
                            obs_tensor = torch.tensor(observations[agent_id][None], device=self.device)
                            output = self.bundle.ppo_policies[agent_id].act(obs_tensor, deterministic=True)
                            actions[agent_id] = int(output.action.item())
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
                done = step.done
            reward_sums.append(reward_total)
            if reward_total > 0:
                wins += 1.0
        return {
            "mean_reward": float(np.mean(reward_sums) if reward_sums else 0.0),
            "win_rate": wins / max(1, num_episodes),
        }

    def _collect_rollout(
        self,
        observations: dict[str, np.ndarray],
        infos: dict[str, dict[str, object]],
        global_step: int,
        episodes: int,
        reward_totals: dict[str, float],
        episode_returns: dict[str, float],
    ) -> tuple[dict[str, dict[str, list[Any]]], dict[str, np.ndarray], dict[str, dict[str, object]], int, int, dict[str, float], dict[str, float]]:
        rollout = {
            agent_id: {
                "obs": [],
                "actions": [],
                "log_probs": [],
                "values": [],
                "rewards": [],
                "dones": [],
            }
            for agent_id in self.trainable_agent_ids
        }

        step_count = 0
        while (
            step_count < self.cfg.training.rollout_steps
            and global_step + step_count < self.cfg.training.total_steps
        ):
            actions: dict[str, int] = {}
            train_outputs: dict[str, tuple[Tensor, Tensor, Tensor]] = {}
            with torch.no_grad():
                for agent_id in self.env.agent_ids:
                    if agent_id in self.trainable_agent_ids:
                        obs_tensor = torch.tensor(observations[agent_id][None], device=self.device)
                        output = self.bundle.ppo_policies[agent_id].act(obs_tensor)
                        actions[agent_id] = int(output.action.item())
                        train_outputs[agent_id] = (
                            output.log_prob.squeeze(0),
                            output.value.squeeze(0),
                            output.action.squeeze(0),
                        )
                        rollout[agent_id]["obs"].append(observations[agent_id].copy())
                        rollout[agent_id]["actions"].append(int(output.action.item()))
                        rollout[agent_id]["log_probs"].append(float(output.log_prob.item()))
                        rollout[agent_id]["values"].append(float(output.value.item()))
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

            for agent_id in self.env.agent_ids:
                reward_totals[agent_id] += step.rewards[agent_id]
                episode_returns[agent_id] += step.rewards[agent_id]
            for agent_id in self.trainable_agent_ids:
                rollout[agent_id]["rewards"].append(float(step.rewards[agent_id]))
                rollout[agent_id]["dones"].append(float(step.terminated[agent_id] or step.truncated[agent_id]))

            observations = step.observations
            infos = step.infos
            step_count += 1
            if step.done:
                episodes += 1
                observations = self.env.reset(seed=self.cfg.seed + episodes + global_step + step_count)
                infos = {agent_id: {} for agent_id in self.env.agent_ids}
                episode_returns = {agent_id: 0.0 for agent_id in self.env.agent_ids}

        for agent_id in self.trainable_agent_ids:
            if rollout[agent_id]["obs"]:
                with torch.no_grad():
                    obs_tensor = torch.tensor(observations[agent_id][None], device=self.device)
                    next_value = self.bundle.ppo_policies[agent_id].act(obs_tensor, deterministic=True).value.item()
                rollout[agent_id]["next_value"] = next_value
            else:
                rollout[agent_id]["next_value"] = 0.0
        return rollout, observations, infos, step_count, episodes, reward_totals, episode_returns

    def _update(self, rollout: dict[str, dict[str, list[Any]]]) -> dict[str, float]:
        policy_losses = []
        value_losses = []
        entropies = []
        for agent_id in self.trainable_agent_ids:
            agent_rollout = rollout[agent_id]
            if not agent_rollout["obs"]:
                continue
            advantages, returns = _compute_gae(
                rewards=np.asarray(agent_rollout["rewards"], dtype=np.float32),
                values=np.asarray(agent_rollout["values"], dtype=np.float32),
                dones=np.asarray(agent_rollout["dones"], dtype=np.float32),
                next_value=float(agent_rollout["next_value"]),
                gamma=self.cfg.training.gamma,
                gae_lambda=self.cfg.training.gae_lambda,
            )
            obs_tensor = torch.tensor(np.stack(agent_rollout["obs"]), device=self.device)
            actions_tensor = torch.tensor(agent_rollout["actions"], device=self.device)
            old_log_probs = torch.tensor(agent_rollout["log_probs"], device=self.device)
            returns_tensor = torch.tensor(returns, device=self.device)
            advantages_tensor = torch.tensor(advantages, device=self.device)
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (
                advantages_tensor.std(unbiased=False) + 1e-8
            )
            optimizer = self.optimizers[agent_id]
            agent_policy_losses = []
            agent_value_losses = []
            agent_entropies = []

            indices = torch.randperm(obs_tensor.shape[0], device=self.device)
            for _ in range(self.cfg.training.ppo_epochs):
                for start in range(0, obs_tensor.shape[0], self.cfg.training.minibatch_size):
                    batch_indices = indices[start : start + self.cfg.training.minibatch_size]
                    batch_obs = obs_tensor[batch_indices]
                    batch_actions = actions_tensor[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_returns = returns_tensor[batch_indices]
                    batch_advantages = advantages_tensor[batch_indices]

                    log_prob, entropy, value = self.bundle.ppo_policies[agent_id].evaluate_actions(
                        batch_obs,
                        batch_actions,
                    )
                    ratio = torch.exp(log_prob - batch_old_log_probs)
                    clipped_ratio = torch.clamp(
                        ratio,
                        1.0 - self.cfg.training.ppo_clip,
                        1.0 + self.cfg.training.ppo_clip,
                    )
                    policy_loss = -torch.min(
                        ratio * batch_advantages,
                        clipped_ratio * batch_advantages,
                    ).mean()
                    value_loss = F.mse_loss(value, batch_returns)
                    entropy_mean = entropy.mean()
                    loss = (
                        policy_loss
                        + self.cfg.training.value_coef * value_loss
                        - self.cfg.training.entropy_coef * entropy_mean
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.bundle.ppo_policies[agent_id].parameters(),
                        self.cfg.training.max_grad_norm,
                    )
                    optimizer.step()
                    agent_policy_losses.append(float(policy_loss.item()))
                    agent_value_losses.append(float(value_loss.item()))
                    agent_entropies.append(float(entropy_mean.item()))

            policy_losses.append(float(np.mean(agent_policy_losses)))
            value_losses.append(float(np.mean(agent_value_losses)))
            entropies.append(float(np.mean(agent_entropies)))
        return {
            "policy_loss": float(np.mean(policy_losses) if policy_losses else 0.0),
            "value_loss": float(np.mean(value_losses) if value_losses else 0.0),
            "entropy": float(np.mean(entropies) if entropies else 0.0),
        }


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards)
    gae = 0.0
    current_next_value = next_value
    for index in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[index]
        delta = rewards[index] + gamma * current_next_value * nonterminal - values[index]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[index] = gae
        current_next_value = values[index]
    returns = advantages + values
    return advantages, returns
