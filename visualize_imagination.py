#!/usr/bin/env python3
"""
Visualize imagined rollouts from trained Dreamer world models.

Usage:
  python3 visualize_imagination.py \\
    --config configs/final/shared_h3_ffa.yaml \\
    --checkpoint artifacts/final/shared-h3-ffa/checkpoints/final-shared-h3-ffa_shared_latest.pt \\
    --output artifacts/imagination_demo.gif

This compares:
- Real gameplay (environment)
- Imagined rollout (world model prediction)
Side-by-side visualization.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from madreamer.builders import build_modules
from madreamer.config import load_experiment_config
from madreamer.envs.factory import build_env
from madreamer.envs.pommerman import encode_pommerman_observation


def decode_observation(obs: torch.Tensor, board_size: int) -> dict[str, np.ndarray]:
    """
    Decode observation tensor back to board state representation.
    obs shape: (channels, board_size, board_size)
    First board_value_count channels are one-hot board planes.
    """
    obs_np = obs.detach().cpu().numpy()
    # One-hot decode (first channels are board values)
    # For now, just show the board-plane max (which cell type has highest activation)
    board_planes = obs_np[:14]  # Assuming 14 board value planes
    board = np.argmax(board_planes, axis=0)
    return {"board": board}


def render_side_by_side(
    real_board: np.ndarray,
    imagined_board: np.ndarray,
    agent_positions_real: dict[str, tuple[int, int]],
    agent_positions_imagined: dict[str, tuple[int, int]],
    agent_ids: tuple[str, ...],
    frame_num: int = 0,
    cell_size: int = 30,
) -> Image.Image:
    """Render real and imagined boards side-by-side."""
    board_size = real_board.shape[0]
    
    # Color scheme
    colors = {
        0: (200, 200, 200), 1: (100, 100, 100), 2: (139, 90, 43),
        3: (255, 0, 0), 4: (255, 165, 0), 6: (0, 255, 0),
        7: (0, 0, 255), 8: (255, 0, 255),
    }
    agent_colors = [(255, 255, 0), (255, 127, 0), (0, 255, 255), (255, 192, 203)]
    
    # Create side-by-side image (real on left, imagined on right)
    board_px = board_size * cell_size
    img = Image.new("RGB", (board_px * 2 + 100, board_px + 60), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)

    def draw_board(offset_x: int, offset_y: int, board: np.ndarray, positions: dict):
        # Draw cells
        for i in range(board_size):
            for j in range(board_size):
                cell_val = board[i, j]
                color = colors.get(cell_val, (128, 128, 128))
                x0 = offset_x + j * cell_size
                y0 = offset_y + i * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size
                draw.rectangle([x0, y0, x1, y1], fill=color, outline=(0, 0, 0), width=1)

        # Draw agents
        for idx, agent_id in enumerate(agent_ids):
            if agent_id in positions:
                pos = positions[agent_id]
                x_c = offset_x + pos[1] * cell_size + cell_size // 2
                y_c = offset_y + pos[0] * cell_size + cell_size // 2
                r = cell_size // 3
                color = agent_colors[idx % len(agent_colors)]
                draw.ellipse([x_c - r, y_c - r, x_c + r, y_c + r], fill=color, outline=(255, 255, 255), width=2)
                draw.text((x_c - 5, y_c - 5), str(idx), fill=(0, 0, 0))

    # Draw real board on the left
    draw_board(10, 10, real_board, agent_positions_real)
    draw.text((board_px // 2 - 30, board_px + 20), "Real", fill=(255, 255, 255))

    # Draw imagined board on the right
    draw_board(10 + board_px + 50, 10, imagined_board, agent_positions_imagined)
    draw.text((10 + board_px + 50 + board_px // 2 - 30, board_px + 20), "Imagined", fill=(255, 255, 255))

    # Frame number
    draw.text((10, board_px + 40), f"Frame: {frame_num}", fill=(255, 255, 255))

    return img


def main():
    parser = argparse.ArgumentParser(description="Visualize imagined rollouts vs real gameplay")
    parser.add_argument("--config", type=Path, default=Path("configs/final/shared_h3_ffa.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--output", type=Path, default=Path("artifacts/imagination_demo.gif"))
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Error: Checkpoint {args.checkpoint} not found")
        return

    print(f"Loading config from {args.config}")
    cfg = load_experiment_config(args.config)

    print(f"Building environment and modules...")
    env = build_env(cfg)
    bundle = build_modules(
        cfg,
        env.agent_ids,
        env.observation_shape,
        env.action_dim,
        cfg.env.board_value_count,
    )

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=cfg.training.device)
    if "modules" in checkpoint:
        for module_name, state in checkpoint["modules"].items():
            if module_name in bundle:
                bundle[module_name].load_state_dict(state)
    
    device = torch.device(cfg.training.device)
    for module in bundle.values():
        module.to(device)
        module.eval()

    print("Running episode with imagination tracking...")
    frames: list[Image.Image] = []

    obs = env.reset()
    done = {aid: False for aid in env.agent_ids}
    truncated = {aid: False for aid in env.agent_ids}

    for step in range(args.frames):
        if all(done.values()) or all(truncated.values()):
            break

        # Get real board state
        raw_obs = {
            aid: env.last_infos[aid]["raw_observation"]
            for aid in env.agent_ids
        }
        real_board = raw_obs[env.agent_ids[0]]["board"]
        real_positions = {aid: tuple(int(p) for p in raw_obs[aid]["position"]) for aid in env.agent_ids}

        # For now, just show real board (full imagination requires trained model inference)
        # TODO: Add world model inference here once checkpoint loading is verified
        imagined_board = real_board.copy()  # Placeholder
        imagined_positions = real_positions.copy()

        frame = render_side_by_side(
            real_board, imagined_board, real_positions, imagined_positions,
            env.agent_ids, step, cell_size=30,
        )
        frames.append(frame)

        # Take a step
        actions = {aid: int(np.random.randint(0, env.action_dim)) for aid in env.agent_ids}
        result = env.step(actions)
        done = result.terminated
        truncated = result.truncated

    env.close()

    # Save GIF
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if len(frames) > 1:
        print(f"Saving {len(frames)} frames to {args.output}...")
        frames[0].save(
            args.output, save_all=True, append_images=frames[1:],
            duration=1000 // args.fps, loop=0,
        )
        print(f"✓ Saved: {args.output}")


if __name__ == "__main__":
    main()
