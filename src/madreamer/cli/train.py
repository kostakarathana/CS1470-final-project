from __future__ import annotations

import argparse
import json
from pathlib import Path

from madreamer.config import load_experiment_config
from madreamer.experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a smoke rollout for a multi-agent RL experiment.")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--steps", type=int, default=None, help="Override training.total_steps")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint to resume from")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_experiment_config(args.config)
    if args.steps is not None:
        cfg.training.total_steps = args.steps
    summary = run_experiment(cfg, checkpoint_path=str(args.checkpoint) if args.checkpoint else None)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
