from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from madreamer.builders import ModuleBundle
from madreamer.config import ExperimentConfig
from madreamer.envs.base import MultiAgentEnv
from madreamer.replay import MultiAgentReplayBuffer, Transition


@dataclass
class RolloutSummary:
    collected_steps: int
    episodes: int
    reward_totals: dict[str, float]
    strategy: str


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

    def run(self) -> RolloutSummary:
        collected_steps = 0
        episodes = 0
        reward_totals = {agent_id: 0.0 for agent_id in self.env.agent_ids}

        while collected_steps < self.cfg.training.total_steps:
            episodes += 1
            observations = self.env.reset(seed=self.cfg.seed + episodes)
            hidden = {
                agent_id: self.bundle.world_models[agent_id].initial_state(1, self.device)
                for agent_id in self.env.agent_ids
            }
            prev_actions = {agent_id: 0 for agent_id in self.env.agent_ids}
            episode_done = False

            while not episode_done and collected_steps < self.cfg.training.total_steps:
                actions: dict[str, int] = {}
                with torch.no_grad():
                    for agent_id in self.env.agent_ids:
                        obs_tensor = torch.tensor(observations[agent_id][None], device=self.device)
                        action_tensor = torch.tensor([prev_actions[agent_id]], device=self.device)
                        opponent_tensor = self._build_opponent_tensor(agent_id, prev_actions)
                        model_output = self.bundle.world_models[agent_id](
                            obs_tensor,
                            action_tensor,
                            hidden[agent_id],
                            opponent_action=opponent_tensor,
                        )
                        hidden[agent_id] = model_output.hidden
                        sampled_action = self.bundle.actors[agent_id].sample_action(model_output.latent)
                        actions[agent_id] = int(sampled_action.item())

                step = self.env.step(actions)
                self.replay.add(
                    Transition(
                        observations=observations,
                        actions=actions,
                        rewards=step.rewards,
                        next_observations=step.observations,
                        terminated=step.terminated,
                        truncated=step.truncated,
                    )
                )
                for agent_id, reward in step.rewards.items():
                    reward_totals[agent_id] += reward
                observations = step.observations
                prev_actions = actions
                collected_steps += 1
                episode_done = step.done

        return RolloutSummary(
            collected_steps=collected_steps,
            episodes=episodes,
            reward_totals=reward_totals,
            strategy=self.cfg.algorithm.name,
        )

    def _build_opponent_tensor(
        self,
        agent_id: str,
        previous_actions: dict[str, int],
    ) -> torch.Tensor | None:
        world_model = self.bundle.world_models[agent_id]
        if not world_model.opponent_action_dim:
            return None
        opponent_ids = [other_id for other_id in self.env.agent_ids if other_id != agent_id]
        one_hots = []
        for other_id in opponent_ids:
            action = torch.tensor(previous_actions[other_id], device=self.device)
            one_hots.append(F.one_hot(action, num_classes=self.env.action_dim).float())
        concatenated = torch.cat(one_hots, dim=0).unsqueeze(0)
        return concatenated
