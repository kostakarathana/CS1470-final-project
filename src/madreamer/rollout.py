from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from madreamer.envs.base import MultiAgentEnv
from madreamer.replay import MultiAgentReplayBuffer, ReplayStep, build_opponent_context

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
    episode_id: int = 0,
) -> EpisodeSummary:
    observations = env.reset(seed=seed)
    steps = 0
    done = False
    total_rewards = {agent_id: 0.0 for agent_id in env.agent_ids}
    infos = env.last_infos

    while not done and (max_steps is None or steps < max_steps):
        actions = {
            agent_id: int(policies[agent_id](agent_id, observations[agent_id], infos.get(agent_id, {})))
            for agent_id in env.agent_ids
        }
        step = env.step(actions)
        if replay is not None:
            replay.add(
                ReplayStep(
                    episode_id=episode_id,
                    observations={agent_id: obs.copy() for agent_id, obs in observations.items()},
                    actions=actions.copy(),
                    opponent_actions={
                        agent_id: build_opponent_context(
                            agent_id,
                            env.agent_ids,
                            actions,
                            step.alive,
                            env.action_dim,
                        )
                        for agent_id in env.agent_ids
                    },
                    rewards=step.rewards.copy(),
                    raw_rewards=step.raw_rewards.copy(),
                    next_observations={
                        agent_id: obs.copy() for agent_id, obs in step.observations.items()
                    },
                    terminated=step.terminated.copy(),
                    truncated=step.truncated.copy(),
                    alive=step.alive.copy(),
                    infos={agent_id: dict(info) for agent_id, info in step.infos.items()},
                    events={agent_id: dict(event) for agent_id, event in step.events.items()},
                )
            )
        for agent_id, reward in step.rewards.items():
            total_rewards[agent_id] += reward
        observations = step.observations
        infos = step.infos
        steps += 1
        done = step.done

    return EpisodeSummary(steps=steps, total_rewards=total_rewards, done=done)
