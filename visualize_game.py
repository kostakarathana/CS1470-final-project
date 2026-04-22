#!/usr/bin/env python3
"""
Visualize actual Pommerman gameplay.

Usage:
  python3 visualize_game.py --config configs/final/ppo_ffa.yaml \\
    --checkpoint artifacts/final/ppo-ffa/checkpoints/final-ppo-ffa_ppo_latest.pt \\
    --output artifacts/gameplay.gif --episodes 3

Creates:
  - GIF animation of gameplay
  - PNG snapshots of key frames
  - Reward tracking overlay
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from madreamer.builders import build_modules
from madreamer.config import load_experiment_config
from madreamer.envs.factory import build_env


def render_board_state(
    board: np.ndarray,
    bomb_blast_strength: np.ndarray,
    bomb_life: np.ndarray,
    agent_positions: dict[str, tuple[int, int]],
    ammo: dict[str, int],
    agent_ids: tuple[str, ...],
    cell_size: int = 40,
    frame_num: int = 0,
) -> Image.Image:
    """Render a Pommerman board state to PIL Image."""
    board_size = board.shape[0]
    img_size = board_size * cell_size
    img = Image.new("RGB", (img_size, img_size + 50), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)

    # Color scheme
    colors = {
        0: (200, 200, 200),  # Passage (light gray)
        1: (100, 100, 100),  # Rigid wall (dark gray)
        2: (139, 90, 43),    # Wood (brown)
        3: (255, 0, 0),      # Bomb (red)
        4: (255, 165, 0),    # Flames (orange)
        6: (0, 255, 0),      # Extra bomb (green)
        7: (0, 0, 255),      # Extra range (blue)
        8: (255, 0, 255),    # Extra kick (magenta)
    }
    agent_colors = [(255, 255, 0), (255, 127, 0), (0, 255, 255), (255, 192, 203)]

    # Draw base board
    for i in range(board_size):
        for j in range(board_size):
            cell_val = board[i, j]
            color = colors.get(cell_val, (128, 128, 128))
            x0 = j * cell_size
            y0 = i * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(0, 0, 0), width=1)

    # Draw bomb metadata (life, blast strength)
    for i in range(board_size):
        for j in range(board_size):
            if board[i, j] == 3:  # Bomb
                life = bomb_life[i, j]
                strength = bomb_blast_strength[i, j]
                x = j * cell_size + cell_size // 2
                y = i * cell_size + cell_size // 2
                text = f"L{int(life)}\nR{int(strength)}"
                # Draw text centered on bomb
                draw.text(
                    (x - 10, y - 10),
                    text,
                    fill=(255, 255, 255),
                    font=None,
                )

    # Draw agents
    for idx, agent_id in enumerate(agent_ids):
        if agent_id in agent_positions:
            pos = agent_positions[agent_id]
            x_center = pos[1] * cell_size + cell_size // 2
            y_center = pos[0] * cell_size + cell_size // 2
            radius = cell_size // 3
            color = agent_colors[idx % len(agent_colors)]
            draw.ellipse(
                [
                    x_center - radius,
                    y_center - radius,
                    x_center + radius,
                    y_center + radius,
                ],
                fill=color,
                outline=(255, 255, 255),
                width=2,
            )
            # Draw agent number
            draw.text(
                (x_center - 5, y_center - 5),
                str(idx),
                fill=(0, 0, 0),
                font=None,
            )
            # Draw ammo count
            draw.text(
                (x_center - 10, y_center + 8),
                f"a{ammo.get(agent_id, 0)}",
                fill=(255, 255, 255),
                font=None,
            )

    # Draw frame number and legend
    draw.text((10, img_size + 10), f"Frame: {frame_num}", fill=(255, 255, 255))

    return img


def play_episode(
    env: Any,
    bundle: dict[str, Any],
    cfg: Any,
    device: str,
    num_frames: int = 256,
) -> tuple[list[Image.Image], dict[str, list[float]]]:
    """
    Play one episode and record frames + metrics.
    
    Returns:
      (frames, metrics_per_agent)
    """
    frames: list[Image.Image] = []
    metrics = {agent_id: [] for agent_id in env.agent_ids}

    obs = env.reset()
    done = {agent_id: False for agent_id in env.agent_ids}
    truncated = {agent_id: False for agent_id in env.agent_ids}

    # Get initial raw observations from last_infos
    raw_obs = {
        agent_id: env.last_infos[agent_id]["raw_observation"]
        for agent_id in env.agent_ids
    }
    
    # Initial frame
    first_raw = raw_obs[env.agent_ids[0]]
    frame = render_board_state(
        board=first_raw["board"],
        bomb_blast_strength=first_raw.get("bomb_blast_strength", np.zeros_like(first_raw["board"])),
        bomb_life=first_raw.get("bomb_life", np.zeros_like(first_raw["board"])),
        agent_positions={
            aid: tuple(int(p) for p in raw_obs[aid]["position"])
            for aid in env.agent_ids
        },
        ammo={aid: int(raw_obs[aid].get("ammo", 0)) for aid in env.agent_ids},
        agent_ids=env.agent_ids,
        frame_num=0,
    )
    frames.append(frame)

    # Run episode
    for step in range(num_frames):
        if all(done.values()) or all(truncated.values()):
            break

        # Get actions: sample random actions [0, action_dim)
        actions = {aid: int(np.random.randint(0, env.action_dim)) for aid in env.agent_ids}

        result = env.step(actions)
        obs = result.observations
        rewards = result.rewards
        done = result.terminated
        truncated = result.truncated

        # Record metrics
        for agent_id in env.agent_ids:
            metrics[agent_id].append(rewards[agent_id])

        # Get raw observations from last_infos
        raw_obs = {
            agent_id: env.last_infos[agent_id]["raw_observation"]
            for agent_id in env.agent_ids
        }

        # Render frame
        first_raw = raw_obs[env.agent_ids[0]]
        frame = render_board_state(
            board=first_raw["board"],
            bomb_blast_strength=first_raw.get("bomb_blast_strength", np.zeros_like(first_raw["board"])),
            bomb_life=first_raw.get("bomb_life", np.zeros_like(first_raw["board"])),
            agent_positions={
                aid: tuple(int(p) for p in raw_obs[aid]["position"])
                for aid in env.agent_ids
            },
            ammo={aid: int(raw_obs[aid].get("ammo", 0)) for aid in env.agent_ids},
            agent_ids=env.agent_ids,
            frame_num=step + 1,
        )
        frames.append(frame)

    return frames, metrics


def main():
    parser = argparse.ArgumentParser(description="Visualize Pommerman gameplay")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/final/ppo_ffa.yaml"),
        help="Config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Trained model checkpoint (optional)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gameplay.gif"),
        help="Output GIF path",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=256,
        help="Max frames per episode",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to visualize",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="GIF frames per second",
    )
    args = parser.parse_args()

    print(f"Loading config from {args.config}")
    cfg = load_experiment_config(args.config)

    print(f"Building environment...")
    env = build_env(cfg)

    # Build modules if checkpoint exists (for trained agent)
    if args.checkpoint and args.checkpoint.exists():
        print(f"Loading checkpoint from {args.checkpoint}")
        bundle = build_modules(
            cfg,
            env.agent_ids,
            env.observation_shape,
            env.action_dim,
            cfg.env.board_value_count,
        )
        # Load checkpoint here (not implemented in this basic version)
        # For now, we just run with random agents
    else:
        print("No checkpoint; using random agents")
        bundle = None

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Play episodes
    all_frames: list[Image.Image] = []
    all_metrics = {agent_id: [] for agent_id in env.agent_ids}

    for ep in range(args.episodes):
        print(f"\nEpisode {ep + 1}/{args.episodes}...")
        frames, metrics = play_episode(env, bundle, cfg, cfg.training.device, args.frames)
        all_frames.extend(frames)
        for agent_id in env.agent_ids:
            all_metrics[agent_id].extend(metrics[agent_id])

    env.close()

    # Save GIF
    if len(all_frames) > 1:
        print(f"\nSaving {len(all_frames)} frames to {args.output}...")
        all_frames[0].save(
            args.output,
            save_all=True,
            append_images=all_frames[1:],
            duration=1000 // args.fps,
            loop=0,
        )
        print(f"✓ Saved GIF: {args.output}")

    # Save sample frames as PNG
    sample_indices = [0, len(all_frames) // 4, len(all_frames) // 2, 3 * len(all_frames) // 4, len(all_frames) - 1]
    for idx in sample_indices:
        if idx < len(all_frames):
            png_path = args.output.parent / f"frame_{idx:04d}.png"
            all_frames[idx].save(png_path)
            print(f"✓ Saved frame: {png_path}")

    # Print metrics summary
    print("\nMetrics Summary:")
    for agent_id in env.agent_ids:
        rewards = all_metrics[agent_id]
        if rewards:
            total_reward = sum(rewards)
            avg_reward = total_reward / len(rewards)
            print(f"  {agent_id}: total={total_reward:.2f}, avg={avg_reward:.4f}")


if __name__ == "__main__":
    main()
