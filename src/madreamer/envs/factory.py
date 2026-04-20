from __future__ import annotations

from collections.abc import Callable

from madreamer.config import ExperimentConfig
from madreamer.envs.mock_grid import MockGridEnv
from madreamer.envs.pommerman import PommermanBackend, PommermanEnv


def build_env(
    cfg: ExperimentConfig,
    pommerman_backend_factory: Callable[[PommermanEnv], PommermanBackend] | None = None,
) -> MockGridEnv | PommermanEnv:
    if cfg.env.name == "mock_grid":
        return MockGridEnv(
            num_agents=cfg.env.num_agents,
            grid_size=cfg.env.grid_size,
            max_steps=cfg.env.max_steps,
            task_type=cfg.env.task_type,
        )

    if cfg.env.name == "pommerman":
        return PommermanEnv(
            num_agents=cfg.env.num_agents,
            env_id=cfg.env.env_id,
            board_size=cfg.env.board_size,
            max_steps=cfg.env.max_steps,
            observability=cfg.env.observability,
            communication=cfg.env.communication,
            board_value_count=cfg.env.board_value_count,
            opponent_mode=cfg.env.opponent_mode,
            backend_factory=pommerman_backend_factory,
        )

    raise ValueError(f"Unknown environment '{cfg.env.name}'.")
