# CS1470 Final Project

PyTorch research codebase for proposal-faithful multi-agent Dreamer and PPO experiments in Pommerman.

The repository is organized around four experiment families:

- `independent`: each agent has its own world model and policy/value heads
- `shared`: agents share one world model but keep separate policy/value heads
- `opponent_aware`: each agent has its own world model conditioned on other agents' actions
- `ppo`: model-free baseline with one PPO-style policy/value network per agent

The codebase now provides:

- config-driven PPO and Dreamer-lite training entrypoints
- a common multi-agent environment interface with reward presets
- Pommerman FFA and Team adapters plus a tiny mock-grid smoke environment
- sequence replay for recurrent world-model training
- checkpointing, JSONL metric logging, evaluation, and analysis helpers

## Quick Start

```bash
./scripts/bootstrap_runtime.sh
python3 -m madreamer.cli.train --config configs/ppo_smoke.yaml --steps 64
python3 -m madreamer.cli.train --config configs/shared_smoke.yaml --steps 4
python3 -m madreamer.cli.eval --config configs/shared_smoke.yaml --checkpoint artifacts/checkpoints/shared-smoke_shared_latest.pt
python3 analyze_results.py
pytest
```

The repo includes a bundled Pommerman source tree under `third_party/pommerman`. The adapter prefers the bundled backend automatically and installs lightweight compatibility stubs for old `rapidjson`, graphics, network, and CLI imports so headless training can run on modern Python without installing the official playground package.

## Config Families

- `*_smoke.yaml`: tiny fake-backend friendly runs
- `*_dev.yaml`: modest local runs for iteration
- `*_study.yaml`: longer proposal-aligned Pommerman experiments
- `configs/final/*.yaml`: final matrix configs for PPO, sharing-strategy, horizon, and team ablations

## Final Matrix and Visualizations

```bash
./scripts/validate_project.sh
./run_matrix.sh
python3 analyze_results.py
python3 visualize_game.py --config configs/final/ppo_ffa.yaml \
  --checkpoint artifacts/final/ppo-ffa/checkpoints/final-ppo-ffa_ppo_latest.pt \
  --output artifacts/poster_gameplay.gif --frames 64
python3 compare_strategies.py --configs ppo_ffa shared_h3_ffa opponent_aware_h3_ffa \
  --checkpoints artifacts/final/ppo-ffa/checkpoints/final-ppo-ffa_ppo_latest.pt \
  artifacts/final/shared-h3-ffa/checkpoints/final-shared-h3-ffa_shared_latest.pt \
  artifacts/final/opponent-aware-h3-ffa/checkpoints/final-opponent-aware-h3-ffa_opponent_aware_latest.pt \
  --titles PPO Shared Opponent-Aware --output artifacts/strategy_comparison.png
```

`analyze_results.py` writes ignored scratch copies under `artifacts/` and commit-friendly copies under `results/`.
