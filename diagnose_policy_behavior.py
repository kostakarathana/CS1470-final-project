#!/usr/bin/env python3
"""Diagnose whether a trained policy is winning or gaming shaped rewards."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from madreamer.config import load_experiment_config
from madreamer.envs.factory import build_env
from visualize_game import build_policy_controller


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose policy outcomes, actions, and event rates.")
    parser.add_argument("--config", type=Path, required=True, help="Experiment config path.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--episodes", type=int, default=64, help="Number of episodes to run.")
    parser.add_argument("--seed-offset", type=int, default=30_000, help="Offset added to config seed.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = diagnose_policy(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        seed_offset=args.seed_offset,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


def diagnose_policy(
    *,
    config_path: Path,
    checkpoint_path: Path,
    episodes: int,
    seed_offset: int,
) -> dict[str, Any]:
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    cfg = load_experiment_config(config_path)
    env = build_env(cfg)
    controller = build_policy_controller(env, cfg, checkpoint_path)
    controlled_agent_ids = controller.controlled_agent_ids or env.agent_ids

    wins = 0
    losses = 0
    ties = 0
    episode_lengths: list[int] = []
    shaped_returns: list[float] = []
    raw_terminal_rewards: list[float] = []
    action_counts: dict[str, Counter[int]] = {agent_id: Counter() for agent_id in controlled_agent_ids}
    event_totals: dict[str, defaultdict[str, float]] = {
        agent_id: defaultdict(float)
        for agent_id in controlled_agent_ids
    }

    for episode in range(episodes):
        observations = env.reset(seed=cfg.seed + seed_offset + episode)
        controller.reset_episode()
        infos = env.last_infos
        done = False
        episode_length = 0
        episode_shaped_return = {agent_id: 0.0 for agent_id in controlled_agent_ids}
        final_raw_rewards = {agent_id: 0.0 for agent_id in controlled_agent_ids}

        while not done:
            actions = controller.actions(observations, infos)
            for agent_id in controlled_agent_ids:
                action_counts[agent_id][int(actions[agent_id])] += 1

            step = env.step(actions)
            controller.after_step(actions, step.alive)

            for agent_id in controlled_agent_ids:
                episode_shaped_return[agent_id] += float(step.rewards[agent_id])
                final_raw_rewards[agent_id] = float(step.raw_rewards[agent_id])
                for event_name, value in step.events[agent_id].items():
                    event_totals[agent_id][event_name] += float(value)

            observations = step.observations
            infos = step.infos
            episode_length += 1
            done = step.done

        raw_terminal_reward = float(np.mean(list(final_raw_rewards.values())))
        raw_terminal_rewards.append(raw_terminal_reward)
        shaped_returns.append(float(np.mean(list(episode_shaped_return.values()))))
        episode_lengths.append(episode_length)
        if any(reward > 0.0 for reward in final_raw_rewards.values()):
            wins += 1
        elif any(reward < 0.0 for reward in final_raw_rewards.values()):
            losses += 1
        else:
            ties += 1

    env.close()
    total_steps = sum(episode_lengths)
    return {
        "algorithm": cfg.algorithm.name,
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "controlled_agent_ids": list(controlled_agent_ids),
        "episodes": float(episodes),
        "wins": float(wins),
        "losses": float(losses),
        "ties": float(ties),
        "win_rate": float(wins / episodes),
        "loss_rate": float(losses / episodes),
        "tie_rate": float(ties / episodes),
        "mean_episode_length": _mean(episode_lengths),
        "mean_shaped_return": _mean(shaped_returns),
        "mean_raw_terminal_reward": _mean(raw_terminal_rewards),
        "action_distribution": {
            agent_id: _action_distribution(action_counts[agent_id], env.action_dim)
            for agent_id in controlled_agent_ids
        },
        "event_rates_per_step": {
            agent_id: {
                event_name: float(total / max(total_steps, 1))
                for event_name, total in sorted(event_totals[agent_id].items())
            }
            for agent_id in controlled_agent_ids
        },
    }


def _action_distribution(counts: Counter[int], action_dim: int) -> dict[str, float]:
    total = sum(counts.values())
    return {
        f"action_{action}": float(counts.get(action, 0) / max(total, 1))
        for action in range(action_dim)
    }


def _mean(values: list[float] | list[int]) -> float:
    return float(np.mean(values)) if values else 0.0


if __name__ == "__main__":
    main()
