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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from madreamer.builders import ModuleBundle, build_modules, move_bundle_to_device
from madreamer.config import load_experiment_config
from madreamer.envs.factory import build_env
from madreamer.models.world_model import RSSMState
from madreamer.opponents import FixedOpponentManager
from madreamer.replay import build_opponent_context


@dataclass
class PolicyController:
    env: Any
    cfg: Any
    bundle: ModuleBundle | None
    device: torch.device
    controlled_agent_ids: tuple[str, ...]
    opponents: FixedOpponentManager
    world_states: dict[str, RSSMState] = field(default_factory=dict)
    prev_actions: dict[str, int] = field(default_factory=dict)
    prev_opponent_contexts: dict[str, np.ndarray] = field(default_factory=dict)

    def reset_episode(self) -> None:
        self.prev_actions = {agent_id: 0 for agent_id in self.env.agent_ids}
        self.prev_opponent_contexts = {
            agent_id: np.zeros(self._opponent_dim(agent_id), dtype=np.float32)
            for agent_id in self.env.agent_ids
        }
        self.world_states = {}
        if self.bundle is not None and self.cfg.algorithm.name != "ppo":
            self.world_states = {
                agent_id: self.bundle.world_models[agent_id].initial_state(1, self.device)
                for agent_id in self.controlled_agent_ids
            }

    def actions(self, observations: dict[str, np.ndarray], infos: dict[str, dict[str, object]]) -> dict[str, int]:
        actions = self.opponents.actions(observations, infos)
        if self.bundle is None:
            return actions
        with torch.no_grad():
            for agent_id in self.controlled_agent_ids:
                obs_tensor = torch.as_tensor(observations[agent_id][None], device=self.device, dtype=torch.float32)
                if self.cfg.algorithm.name == "ppo":
                    output = self.bundle.ppo_policies[agent_id].act(obs_tensor, deterministic=True)
                    actions[agent_id] = int(output.action.item())
                    continue

                world_model = self.bundle.world_models[agent_id]
                prev_action = torch.as_tensor([self.prev_actions[agent_id]], device=self.device, dtype=torch.long)
                opponent_context = self.prev_opponent_contexts[agent_id]
                opponent_tensor = (
                    torch.as_tensor(opponent_context[None], device=self.device, dtype=torch.float32)
                    if world_model.opponent_action_dim
                    else None
                )
                observed = world_model.observe(
                    obs_tensor,
                    prev_action,
                    self.world_states[agent_id],
                    opponent_tensor,
                    deterministic=True,
                )
                self.world_states[agent_id] = _detach_state(observed.posterior_state)
                actor_output = self.bundle.actors[agent_id].act(
                    observed.posterior_state.features,
                    deterministic=True,
                )
                actions[agent_id] = int(actor_output.action.item())
        return actions

    def after_step(self, actions: dict[str, int], alive: dict[str, bool]) -> None:
        self.prev_actions = actions.copy()
        self.prev_opponent_contexts = {
            agent_id: build_opponent_context(
                agent_id,
                self.env.agent_ids,
                actions,
                alive,
                self.env.action_dim,
            )
            for agent_id in self.env.agent_ids
        }

    def _opponent_dim(self, agent_id: str) -> int:
        if self.bundle is None or self.cfg.algorithm.name == "ppo" or agent_id not in self.bundle.world_models:
            return 0
        return self.bundle.world_models[agent_id].opponent_action_dim


def build_policy_controller(env: Any, cfg: Any, checkpoint: Path | None) -> PolicyController:
    device = torch.device(cfg.training.device)
    bundle: ModuleBundle | None = None
    controlled_agent_ids: tuple[str, ...] = ()
    if checkpoint is not None:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        bundle = build_modules(
            cfg,
            env.agent_ids,
            env.observation_shape,
            env.action_dim,
            cfg.env.board_value_count,
        )
        move_bundle_to_device(bundle, cfg.training.device)
        _load_checkpoint_into_bundle(checkpoint, bundle, device)
        _set_bundle_eval(bundle)
        controlled_agent_ids = (
            (env.agent_ids[0],)
            if cfg.algorithm.learner_setup == "single_learner"
            else env.agent_ids
        )
    controller = PolicyController(
        env=env,
        cfg=cfg,
        bundle=bundle,
        device=device,
        controlled_agent_ids=controlled_agent_ids,
        opponents=FixedOpponentManager(
            env,
            policy_name=cfg.algorithm.opponent_policy,
            controlled_agent_ids=controlled_agent_ids,
            seed=cfg.seed + 50_000,
        ),
    )
    controller.reset_episode()
    return controller


def _load_checkpoint_into_bundle(path: Path, bundle: ModuleBundle, device: torch.device) -> None:
    payload = torch.load(path, map_location=device)
    state = payload.get("bundle", payload)
    loaded = 0
    for agent_id, state_dict in state.get("world_models", {}).items():
        if agent_id in bundle.world_models:
            bundle.world_models[agent_id].load_state_dict(state_dict)
            loaded += 1
    for agent_id, state_dict in state.get("actors", {}).items():
        if agent_id in bundle.actors:
            bundle.actors[agent_id].load_state_dict(state_dict)
            loaded += 1
    for agent_id, state_dict in state.get("critics", {}).items():
        if agent_id in bundle.critics:
            bundle.critics[agent_id].load_state_dict(state_dict)
            loaded += 1
    for agent_id, state_dict in state.get("ppo_policies", {}).items():
        if agent_id in bundle.ppo_policies:
            bundle.ppo_policies[agent_id].load_state_dict(state_dict)
            loaded += 1
    if loaded == 0:
        raise ValueError(f"Checkpoint {path} did not contain any compatible module weights.")


def _set_bundle_eval(bundle: ModuleBundle) -> None:
    seen: set[int] = set()
    for module_group in (bundle.world_models, bundle.actors, bundle.critics, bundle.ppo_policies):
        for module in module_group.values():
            if id(module) in seen:
                continue
            seen.add(id(module))
            module.eval()


def _detach_state(state: RSSMState) -> RSSMState:
    return RSSMState(
        deter=state.deter.detach(),
        stoch=state.stoch.detach(),
        mean=state.mean.detach(),
        std=state.std.detach(),
    )


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
    controller: PolicyController,
    cfg: Any,
    device: str,
    num_frames: int = 256,
    seed: int | None = None,
) -> tuple[list[Image.Image], dict[str, list[float]]]:
    """
    Play one episode and record frames + metrics.
    
    Returns:
      (frames, metrics_per_agent)
    """
    frames: list[Image.Image] = []
    metrics = {agent_id: [] for agent_id in env.agent_ids}

    obs = env.reset(seed=cfg.seed if seed is None else seed)
    controller.reset_episode()
    infos = env.last_infos
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

        actions = controller.actions(obs, infos)

        result = env.step(actions)
        obs = result.observations
        infos = result.infos
        rewards = result.rewards
        done = result.terminated
        truncated = result.truncated
        controller.after_step(actions, result.alive)

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

    print("Building environment...")
    env = build_env(cfg)

    # Build modules if checkpoint exists (for trained agent)
    if args.checkpoint and args.checkpoint.exists():
        print(f"Loading checkpoint from {args.checkpoint}")
        controller = build_policy_controller(env, cfg, args.checkpoint)
    elif args.checkpoint:
        raise FileNotFoundError(args.checkpoint)
    else:
        print("No checkpoint; using configured scripted opponents for all agents")
        controller = build_policy_controller(env, cfg, None)

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Play episodes
    all_frames: list[Image.Image] = []
    all_metrics = {agent_id: [] for agent_id in env.agent_ids}

    for ep in range(args.episodes):
        print(f"\nEpisode {ep + 1}/{args.episodes}...")
        frames, metrics = play_episode(
            env,
            controller,
            cfg,
            cfg.training.device,
            args.frames,
            seed=cfg.seed + ep,
        )
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
