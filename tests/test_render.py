from pathlib import Path

import numpy as np

from madreamer.render import EpisodeFrame, SpriteRenderer, build_mock_grid_render_board


def test_mock_grid_renderer_writes_gif(tmp_path: Path) -> None:
    renderer = SpriteRenderer(tile_size=24)
    obs = np.zeros((3, 5, 5), dtype=np.float32)
    obs[0, 4, 3] = 1.0
    obs[1, 2, 1] = 1.0
    obs[2, 2, 2] = 1.0
    board = build_mock_grid_render_board(obs)
    frames = [
        renderer.render_mock_grid_frame(
            EpisodeFrame(
                step_index=0,
                board=board,
                rewards={"agent_0": 0.0, "agent_1": 0.0},
                actions={"agent_0": 2, "agent_1": 3},
                info_text="test",
            )
        )
    ]
    target = renderer.save_gif(tmp_path / "demo.gif", frames, fps=4.0)
    assert target.exists()


def test_pommerman_renderer_builds_frame() -> None:
    renderer = SpriteRenderer(tile_size=24)
    board = np.zeros((11, 11), dtype=np.int64)
    board[1, 1] = 10
    board[9, 9] = 12
    board[3, 3] = 3
    board[4, 4] = 4
    board[2, 2] = 1
    board[5, 5] = 2
    bomb_life = np.zeros((11, 11), dtype=np.int64)
    bomb_life[3, 3] = 7
    image = renderer.render_pommerman_frame(
        EpisodeFrame(
            step_index=1,
            board=board,
            bomb_life=bomb_life,
            rewards={"agent_0": 0.0},
            actions={"agent_0": 5},
            info_text="pommerman",
        )
    )
    assert image.size[0] > 0
    assert image.size[1] > 0
