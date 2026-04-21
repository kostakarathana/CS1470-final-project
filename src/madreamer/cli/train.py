from __future__ import annotations

import argparse
import json
from pathlib import Path

from madreamer.config import apply_overrides, load_experiment_config
from madreamer.experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a multi-agent Dreamer or PPO experiment.")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--steps", type=int, default=None, help="Override training.total_steps.")
    parser.add_argument("--seed", type=int, default=None, help="Override experiment seed.")
    parser.add_argument("--device", type=str, default=None, help="Override training device.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from.")
    parser.add_argument("--logdir", type=str, default=None, help="Override training.log_dir.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_experiment_config(args.config)
    apply_overrides(
        cfg,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        resume_checkpoint=args.resume,
        log_dir=args.logdir,
    )
    summary = run_experiment(cfg)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
