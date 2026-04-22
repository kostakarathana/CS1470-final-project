#!/usr/bin/env python3
"""
Create a multi-panel visualization comparing strategies side-by-side.

Usage:
  python3 compare_strategies.py \
    --configs ppo_ffa shared_h3_ffa opponent_aware_h3_ffa \
    --checkpoints ppo_latest.pt shared_latest.pt opponent_latest.pt \
    --output artifacts/strategy_comparison.png

Creates a grid showing:
- PPO (baseline) gameplay
- Shared model gameplay  
- Opponent-aware gameplay
Side-by-side for direct visual comparison.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from madreamer.config import load_experiment_config
from madreamer.envs.factory import build_env


def render_compact_board(
    board: np.ndarray,
    agent_positions: dict[str, tuple[int, int]],
    agent_ids: tuple[str, ...],
    cell_size: int = 25,
    title: str = "",
) -> Image.Image:
    """Render a compact board snapshot."""
    board_size = board.shape[0]
    
    colors = {
        0: (200, 200, 200), 1: (100, 100, 100), 2: (139, 90, 43),
        3: (255, 0, 0), 4: (255, 165, 0), 6: (0, 255, 0),
        7: (0, 0, 255), 8: (255, 0, 255),
    }
    agent_colors = [(255, 255, 0), (255, 127, 0), (0, 255, 255), (255, 192, 203)]
    
    board_px = board_size * cell_size
    header = 30 if title else 0
    img = Image.new("RGB", (board_px, board_px + header), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)

    # Title
    if title:
        draw.text((board_px // 2 - 20, 5), title, fill=(255, 255, 255))

    # Draw cells
    for i in range(board_size):
        for j in range(board_size):
            cell_val = board[i, j]
            color = colors.get(cell_val, (128, 128, 128))
            x0 = j * cell_size
            y0 = i * cell_size + header
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle([x0, y0, x1, y1], fill=color, outline=(50, 50, 50), width=1)

    # Draw agents (compact, no labels)
    for idx, agent_id in enumerate(agent_ids):
        if agent_id in agent_positions:
            pos = agent_positions[agent_id]
            x_c = pos[1] * cell_size + cell_size // 2
            y_c = pos[0] * cell_size + cell_size // 2 + header
            r = cell_size // 4
            color = agent_colors[idx % len(agent_colors)]
            draw.ellipse([x_c - r, y_c - r, x_c + r, y_c + r], fill=color, outline=(255, 255, 255), width=1)

    return img


def create_strategy_grid(
    configs: list[str],
    titles: list[str],
    num_snapshots: int = 5,
) -> Image.Image:
    """
    Create a grid comparing multiple strategies.
    Each row is a strategy, each column is a time step.
    """
    rows = len(configs)
    cols = num_snapshots
    board_px = 11 * 25  # 11x11 board, 25px per cell
    
    # Create grid image
    grid_img = Image.new("RGB", (board_px * cols + 50, board_px * rows + 100), color=(20, 20, 20))
    draw = ImageDraw.Draw(grid_img)

    # Title
    draw.text((10, 10), "Strategy Comparison: Pommerman Gameplay Over Time", fill=(255, 255, 255))

    # For now, draw placeholder
    draw.text((10, 50), "(Requires trained models to populate with real gameplay)", fill=(150, 150, 150))
    draw.text((10, 70), f"Expected: {rows} strategies × {cols} time steps", fill=(150, 150, 150))

    return grid_img


def main():
    parser = argparse.ArgumentParser(description="Compare strategies side-by-side")
    parser.add_argument("--configs", type=str, nargs="+", required=True, help="Config names")
    parser.add_argument("--titles", type=str, nargs="+", help="Display titles")
    parser.add_argument("--checkpoints", type=Path, nargs="+", help="Checkpoint paths")
    parser.add_argument("--output", type=Path, default=Path("artifacts/strategy_comparison.png"))
    parser.add_argument("--snapshots", type=int, default=5, help="Time steps to snapshot")
    args = parser.parse_args()

    if args.titles is None:
        args.titles = args.configs

    print(f"Creating comparison grid for: {args.configs}")
    grid = create_strategy_grid(args.configs, args.titles, args.snapshots)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output)
    print(f"✓ Saved: {args.output}")


if __name__ == "__main__":
    main()
