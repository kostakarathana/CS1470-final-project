from __future__ import annotations

import argparse
import json
from pathlib import Path

from madreamer.config import load_experiment_config
from madreamer.experiment import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved multi-agent experiment.")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to load before evaluation.")
    parser.add_argument("--episodes", type=int, default=None, help="Number of evaluation episodes.")
    parser.add_argument("--opponent-policy", type=str, default=None, help="Override evaluation opponent policy.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON path for metrics.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_experiment_config(args.config)
    metrics = run_evaluation(
        cfg,
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        opponent_policy=args.opponent_policy,
    )
    rendered = json.dumps(metrics, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
