from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import torch

from madreamer.builders import build_modules, move_bundle_to_device
from madreamer.checkpoint import load_checkpoint
from madreamer.config import load_experiment_config
from madreamer.envs.factory import build_env
from madreamer.render import EpisodeFrame, SpriteRenderer, build_mock_grid_render_board
from madreamer.scripted import build_scripted_policy, render_mock_grid_observation, sleep_for_render


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a small terminal demo episode.")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint for a trained policy")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--gif", type=Path, default=None, help="Save an animated GIF of the episode")
    parser.add_argument("--fps", type=float, default=5.0, help="GIF playback frames per second")
    parser.add_argument("--open", action="store_true", help="Open the saved GIF after rendering")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_experiment_config(args.config)
    env = build_env(cfg)
    device = torch.device(cfg.training.device)

    trained_agent_ids: tuple[str, ...] = ()
    bundle = None
    hidden = {}
    previous_actions = {agent_id: 0 for agent_id in env.agent_ids}
    if args.checkpoint is not None:
        bundle = build_modules(cfg, env.agent_ids, env.observation_shape, env.action_dim)
        move_bundle_to_device(bundle, cfg.training.device)
        load_checkpoint(args.checkpoint, bundle, device=cfg.training.device)
        trained_agent_ids = (env.agent_ids[0],) if cfg.env.opponent_mode == "fixed" else env.agent_ids
        if cfg.algorithm.name != "ppo":
            hidden = {
                agent_id: bundle.world_models[agent_id].initial_state(1, device)
                for agent_id in trained_agent_ids
            }

    observations = env.reset(seed=cfg.seed)
    infos = {agent_id: {} for agent_id in env.agent_ids}
    renderer = SpriteRenderer()
    frames = []
    policies = {
        agent_id: build_scripted_policy(env, cfg.seed + index, env.action_dim)
        for index, agent_id in enumerate(env.agent_ids)
    }
    for step_index in range(args.steps):
        if cfg.env.name == "mock_grid":
            print(f"\nStep {step_index}")
            print(render_mock_grid_observation(observations[env.agent_ids[0]]))
        elif cfg.env.name == "pommerman" and hasattr(env, "render_text"):
            print(f"\nStep {step_index}")
            print(env.render_text(env.agent_ids[0]))
        actions = {}
        with torch.no_grad():
            for agent_id in env.agent_ids:
                if args.checkpoint is not None and agent_id in trained_agent_ids and bundle is not None:
                    obs_tensor = torch.tensor(observations[agent_id][None], device=device)
                    if cfg.algorithm.name == "ppo":
                        output = bundle.ppo_policies[agent_id].act(obs_tensor, deterministic=True)
                        actions[agent_id] = int(output.action.item())
                    else:
                        prev_action_tensor = torch.tensor([previous_actions[agent_id]], device=device)
                        opponent_tensor = _build_opponent_tensor(
                            agent_id,
                            previous_actions,
                            env.agent_ids,
                            env.action_dim,
                            device,
                            bundle.world_models[agent_id].opponent_action_dim > 0,
                        )
                        observed = bundle.world_models[agent_id].observe(
                            obs_tensor,
                            prev_action_tensor,
                            hidden[agent_id],
                            opponent_action=opponent_tensor,
                        )
                        hidden[agent_id] = observed.hidden
                        logits = bundle.actors[agent_id](observed.latent)
                        actions[agent_id] = int(logits.argmax(dim=-1).item())
                else:
                    actions[agent_id] = policies[agent_id](
                        agent_id,
                        observations[agent_id],
                        infos.get(agent_id, {}),
                    )
        print(f"actions: {actions}")
        step = env.step(actions)
        print(f"rewards: {step.rewards}")
        if args.gif is not None:
            frames.append(_build_frame(renderer, cfg.env.name, env, observations, step_index, actions, step.rewards))
        observations = step.observations
        infos = step.infos
        previous_actions = actions
        sleep_for_render(args.sleep)
        if step.done:
            print("\nEpisode finished.")
            break
    if args.gif is not None and frames:
        output_path = renderer.save_gif(args.gif, frames, fps=args.fps)
        print(f"\nSaved animation to {output_path}")
        if args.open:
            _open_path(output_path)


def _build_opponent_tensor(
    agent_id: str,
    previous_actions: dict[str, int],
    agent_ids: tuple[str, ...],
    action_dim: int,
    device: torch.device,
    enabled: bool,
) -> torch.Tensor | None:
    if not enabled:
        return None
    one_hots = []
    for other_id in agent_ids:
        if other_id == agent_id:
            continue
        action = torch.tensor(previous_actions[other_id], device=device)
        one_hots.append(torch.nn.functional.one_hot(action, num_classes=action_dim).float())
    return torch.cat(one_hots, dim=0).unsqueeze(0)


def _build_frame(
    renderer: SpriteRenderer,
    env_name: str,
    env: object,
    observations: dict[str, object],
    step_index: int,
    actions: dict[str, int],
    rewards: dict[str, float],
):
    if env_name == "pommerman" and hasattr(env, "last_raw_observations"):
        raw = env.last_raw_observations[env.agent_ids[0]]
        board = raw["board"]
        bomb_life = raw.get("bomb_life")
        return renderer.render_pommerman_frame(
            EpisodeFrame(
                step_index=step_index,
                board=board,
                bomb_life=bomb_life,
                rewards=rewards,
                actions=actions,
                info_text="real pommerman episode",
            )
        )
    return renderer.render_mock_grid_frame(
        EpisodeFrame(
            step_index=step_index,
            board=build_mock_grid_render_board(observations[env.agent_ids[0]]),
            rewards=rewards,
            actions=actions,
            info_text="mock grid episode",
        )
    )


def _open_path(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", str(path)], check=False)


if __name__ == "__main__":
    main()
