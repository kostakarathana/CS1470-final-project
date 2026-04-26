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

import numpy as np
from PIL import Image, ImageDraw

from madreamer.config import load_experiment_config
from madreamer.envs.factory import build_env
from visualize_game import build_policy_controller, play_episode


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
    configs: list[Path],
    checkpoints: list[Path | None],
    titles: list[str],
    num_snapshots: int = 5,
) -> Image.Image:
    """
    Create a grid comparing multiple strategies.
    Each row is a strategy, each column is a time step.
    """
    rows = len(configs)
    cols = num_snapshots
    snapshots_by_row: list[list[Image.Image]] = []
    target_size = (275, 320)
    for config_path, checkpoint_path in zip(configs, checkpoints, strict=True):
        cfg = load_experiment_config(config_path)
        env = build_env(cfg)
        controller = build_policy_controller(env, cfg, checkpoint_path)
        frames, _ = play_episode(env, controller, cfg, cfg.training.device, num_frames=max(8, cols * 8))
        env.close()
        indices = _sample_indices(len(frames), cols)
        snapshots_by_row.append([frames[index].convert("RGB").resize(target_size) for index in indices])

    title_height = 48
    row_label_width = 190
    col_header_height = 28
    padding = 12
    canvas_width = row_label_width + cols * target_size[0] + (cols + 1) * padding
    canvas_height = title_height + col_header_height + rows * target_size[1] + (rows + 1) * padding
    grid_img = Image.new("RGB", (canvas_width, canvas_height), color=(20, 20, 20))
    draw = ImageDraw.Draw(grid_img)
    draw.text((padding, 14), "Strategy Comparison: Pommerman Gameplay Over Time", fill=(255, 255, 255))

    for col in range(cols):
        x = row_label_width + padding + col * (target_size[0] + padding)
        draw.text((x + 6, title_height), f"snapshot {col + 1}", fill=(210, 210, 210))
    for row, title in enumerate(titles):
        y = title_height + col_header_height + padding + row * (target_size[1] + padding)
        draw.text((padding, y + 8), title, fill=(255, 255, 255))
        for col, image in enumerate(snapshots_by_row[row]):
            x = row_label_width + padding + col * (target_size[0] + padding)
            grid_img.paste(image, (x, y))
    return grid_img


def _sample_indices(num_frames: int, count: int) -> list[int]:
    if num_frames <= 0:
        raise ValueError("Cannot sample snapshots from an empty episode.")
    if count <= 1:
        return [0]
    return [int(round(value)) for value in np.linspace(0, num_frames - 1, count)]


def _resolve_config(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    if path.suffix != ".yaml":
        path = Path("configs/final") / f"{value}.yaml"
    else:
        path = Path("configs/final") / value
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Compare strategies side-by-side")
    parser.add_argument("--configs", type=str, nargs="+", required=True, help="Config paths or final config names")
    parser.add_argument("--titles", type=str, nargs="+", help="Display titles")
    parser.add_argument("--checkpoints", type=Path, nargs="+", help="Checkpoint paths")
    parser.add_argument("--output", type=Path, default=Path("artifacts/strategy_comparison.png"))
    parser.add_argument("--snapshots", type=int, default=5, help="Time steps to snapshot")
    args = parser.parse_args()

    configs = [_resolve_config(value) for value in args.configs]
    if args.titles is None:
        args.titles = [path.stem for path in configs]
    if len(args.titles) != len(configs):
        raise ValueError("--titles must have the same length as --configs")
    checkpoints = args.checkpoints or [None for _ in configs]
    if len(checkpoints) != len(configs):
        raise ValueError("--checkpoints must have the same length as --configs when provided")

    print(f"Creating comparison grid for: {[str(path) for path in configs]}")
    grid = create_strategy_grid(configs, checkpoints, args.titles, args.snapshots)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output)
    print(f"✓ Saved: {args.output}")


if __name__ == "__main__":
    main()
