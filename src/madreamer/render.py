from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class EpisodeFrame:
    step_index: int
    board: np.ndarray
    rewards: dict[str, float]
    actions: dict[str, int]
    info_text: str
    bomb_life: np.ndarray | None = None


class SpriteRenderer:
    def __init__(self, tile_size: int = 48) -> None:
        self.tile_size = tile_size
        self.resource_dir = Path(__file__).resolve().parents[2] / "third_party" / "pommerman" / "resources"
        self.font_path = self.resource_dir / "Cousine-Regular.ttf"
        self._tile_cache: dict[str, Image.Image] = {}
        self._font_cache: dict[int, ImageFont.ImageFont] = {}

    def render_pommerman_frame(self, frame: EpisodeFrame) -> Image.Image:
        board = np.asarray(frame.board)
        board_size = board.shape[0]
        canvas_width = board_size * self.tile_size
        header_height = 76
        canvas = Image.new("RGBA", (canvas_width, header_height + board_size * self.tile_size), (41, 39, 51, 255))

        for row in range(board_size):
            for col in range(board_size):
                position = (col * self.tile_size, header_height + row * self.tile_size)
                tile = self._build_pommerman_tile(int(board[row, col]), frame.bomb_life, row, col)
                canvas.alpha_composite(tile, position)

        draw = ImageDraw.Draw(canvas)
        title_font = self._font(18)
        body_font = self._font(14)
        draw.text((14, 8), f"Pommerman Step {frame.step_index}", fill=(255, 255, 255, 255), font=title_font)
        draw.text((14, 34), frame.info_text, fill=(220, 220, 220, 255), font=body_font)
        action_text = ", ".join(f"{agent}:{action}" for agent, action in frame.actions.items())
        reward_text = ", ".join(f"{agent}:{reward:.1f}" for agent, reward in frame.rewards.items())
        draw.text((14, 54), f"actions {action_text}", fill=(190, 220, 255, 255), font=body_font)
        draw.text((14, 72 - 18), f"rewards {reward_text}", fill=(200, 255, 200, 255), font=body_font)
        return canvas.convert("P", palette=Image.ADAPTIVE)

    def render_mock_grid_frame(self, frame: EpisodeFrame) -> Image.Image:
        board = np.asarray(frame.board)
        board_size = board.shape[0]
        canvas_width = board_size * self.tile_size
        header_height = 76
        canvas = Image.new("RGBA", (canvas_width, header_height + board_size * self.tile_size), (28, 30, 36, 255))
        colors = {
            0: (42, 47, 59, 255),
            1: (88, 180, 255, 255),
            2: (247, 196, 78, 255),
            3: (225, 111, 111, 255),
        }
        draw = ImageDraw.Draw(canvas)
        for row in range(board_size):
            for col in range(board_size):
                value = int(board[row, col])
                color = colors.get(value, (70, 70, 70, 255))
                x0 = col * self.tile_size
                y0 = header_height + row * self.tile_size
                draw.rectangle((x0, y0, x0 + self.tile_size - 2, y0 + self.tile_size - 2), fill=color)
        title_font = self._font(18)
        body_font = self._font(14)
        draw.text((14, 8), f"MockGrid Step {frame.step_index}", fill=(255, 255, 255, 255), font=title_font)
        draw.text((14, 34), frame.info_text, fill=(220, 220, 220, 255), font=body_font)
        action_text = ", ".join(f"{agent}:{action}" for agent, action in frame.actions.items())
        reward_text = ", ".join(f"{agent}:{reward:.1f}" for agent, reward in frame.rewards.items())
        draw.text((14, 54), f"actions {action_text}", fill=(190, 220, 255, 255), font=body_font)
        draw.text((14, 72 - 18), f"rewards {reward_text}", fill=(200, 255, 200, 255), font=body_font)
        return canvas.convert("P", palette=Image.ADAPTIVE)

    def save_gif(self, path: str | Path, frames: list[Image.Image], fps: float) -> Path:
        if not frames:
            raise ValueError("Cannot save animation without frames.")
        duration_ms = max(20, int(1000 / max(1e-6, fps)))
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            target,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
        return target

    def _build_pommerman_tile(
        self,
        value: int,
        bomb_life: np.ndarray | None,
        row: int,
        col: int,
    ) -> Image.Image:
        base = self._sprite("Passage").copy()
        if value == 1:
            return self._sprite("Rigid")
        if value == 2:
            return self._sprite("Wood")
        if value == 3:
            sprite_name = self._bomb_sprite_name(bomb_life, row, col)
            base.alpha_composite(self._sprite(sprite_name), (0, 0))
            return base
        if value == 4:
            base.alpha_composite(self._sprite("Flames"), (0, 0))
            return base
        if value == 5:
            return self._sprite("Fog")
        if value == 6:
            base.alpha_composite(self._sprite("ExtraBomb"), (0, 0))
            return base
        if value == 7:
            base.alpha_composite(self._sprite("IncrRange"), (0, 0))
            return base
        if value == 8:
            base.alpha_composite(self._sprite("Kick"), (0, 0))
            return base
        if value in {10, 11, 12, 13}:
            agent_name = f"Agent{value - 10}-No-Background"
            base.alpha_composite(self._sprite(agent_name), (0, 0))
            return base
        return base

    def _bomb_sprite_name(self, bomb_life: np.ndarray | None, row: int, col: int) -> str:
        if bomb_life is None:
            return "Bomb"
        life = int(np.asarray(bomb_life)[row, col])
        index = max(1, min(10, life + 1))
        return f"Bomb-{index}"

    def _sprite(self, name: str) -> Image.Image:
        if name not in self._tile_cache:
            path = self.resource_dir / f"{name}.png"
            image = Image.open(path).convert("RGBA").resize((self.tile_size, self.tile_size), Image.Resampling.LANCZOS)
            self._tile_cache[name] = image
        return self._tile_cache[name]

    def _font(self, size: int) -> ImageFont.ImageFont:
        if size not in self._font_cache:
            if self.font_path.exists():
                self._font_cache[size] = ImageFont.truetype(str(self.font_path), size=size)
            else:
                self._font_cache[size] = ImageFont.load_default()
        return self._font_cache[size]


def build_mock_grid_render_board(observation: np.ndarray) -> np.ndarray:
    board = np.zeros((observation.shape[1], observation.shape[2]), dtype=np.int64)
    board[np.argwhere(observation[2] > 0.5)[:, 0], np.argwhere(observation[2] > 0.5)[:, 1]] = 2
    other_positions = np.argwhere(observation[1] > 0.5)
    if len(other_positions):
        board[other_positions[:, 0], other_positions[:, 1]] = 3
    self_positions = np.argwhere(observation[0] > 0.5)
    if len(self_positions):
        board[self_positions[:, 0], self_positions[:, 1]] = 1
    return board
