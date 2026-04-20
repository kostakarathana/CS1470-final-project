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
python3 -m madreamer.cli.eval --config configs/shared_smoke.yaml --checkpoint artifacts/checkpoints/shared-smoke_shared_latest.pt
pytest
```

Note: on this machine, the official playground install hits an upstream `python-rapidjson~=0.6.3` build failure on macOS/Python 3.10. The bootstrap script now reports that explicitly instead of failing silently.

## Config Families

- `*_smoke.yaml`: tiny fake-backend friendly runs
- `*_dev.yaml`: modest local runs for iteration
- `*_study.yaml`: longer proposal-aligned Pommerman experiments
