from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from madreamer.envs.base import MultiAgentEnv
from madreamer.replay import MultiAgentReplayBuffer, Transition

PolicyFn = Callable[[str, np.ndarray, dict[str, object]], int]


@dataclass
class EpisodeSummary:
    steps: int
    total_rewards: dict[str, float]
    done: bool


def collect_episode(
    env: MultiAgentEnv,
    policies: dict[str, PolicyFn],
    *,
    replay: MultiAgentReplayBuffer | None = None,
    seed: int | None = None,
    max_steps: int | None = None,
) -> EpisodeSummary:
    observations = env.reset(seed=seed)
    steps = 0
    done = False
    total_rewards = {agent_id: 0.0 for agent_id in env.agent_ids}
    infos = {agent_id: {} for agent_id in env.agent_ids}

    while not done and (max_steps is None or steps < max_steps):
        actions = {
            agent_id: int(policies[agent_id](agent_id, observations[agent_id], infos.get(agent_id, {})))
            for agent_id in env.agent_ids
        }
        step = env.step(actions)
        if replay is not None:
            replay.add(
                Transition(
                    observations={agent_id: obs.copy() for agent_id, obs in observations.items()},
                    actions=actions.copy(),
                    rewards=step.rewards.copy(),
                    next_observations={
                        agent_id: obs.copy() for agent_id, obs in step.observations.items()
                    },
                    terminated=step.terminated.copy(),
                    truncated=step.truncated.copy(),
                    infos={agent_id: dict(info) for agent_id, info in step.infos.items()},
                )
            )
        for agent_id, reward in step.rewards.items():
            total_rewards[agent_id] += reward
        observations = step.observations
        infos = step.infos
        steps += 1
        done = step.done

    return EpisodeSummary(steps=steps, total_rewards=total_rewards, done=done)
