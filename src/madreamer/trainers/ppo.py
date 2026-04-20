from __future__ import annotations

from dataclasses import dataclass

import torch

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

    def run(self) -> RolloutSummary:
        collected_steps = 0
        episodes = 0
        reward_totals = {agent_id: 0.0 for agent_id in self.env.agent_ids}

        while collected_steps < self.cfg.training.total_steps:
            episodes += 1
            observations = self.env.reset(seed=self.cfg.seed + episodes)
            episode_done = False

            while not episode_done and collected_steps < self.cfg.training.total_steps:
                actions: dict[str, int] = {}
                with torch.no_grad():
                    for agent_id in self.env.agent_ids:
                        obs_tensor = torch.tensor(observations[agent_id][None], device=self.device)
                        output = self.bundle.ppo_policies[agent_id].act(obs_tensor)
                        actions[agent_id] = int(output.action.item())

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
                collected_steps += 1
                episode_done = step.done

        return RolloutSummary(
            collected_steps=collected_steps,
            episodes=episodes,
            reward_totals=reward_totals,
            strategy=self.cfg.algorithm.name,
        )
