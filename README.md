# CS1470 Final Project

At a high level, this project asks whether model-based reinforcement learning can help in a multiplayer setting where the world is changing because other agents are acting too. We use Pommerman, a four-agent grid game with bombs, walls, powerups, and sparse win/loss outcomes, as a compact testbed for comparing a standard model-free learner against agents that learn an internal model of the game and plan through imagined futures.

PyTorch research code for comparing a PPO baseline with Dreamer-style world-model agents in Pommerman. The project focuses on whether learned dynamics models remain useful when other agents make the environment non-stationary.

## What Is Here

- `src/madreamer/`: core package with config loading, environment adapters, replay, PPO, Dreamer-lite trainers, model definitions, rollout helpers, and CLIs.
- `configs/`: experiment definitions. `configs/final/` is the local final matrix, `configs/oscar/` is the longer cluster matrix, and `*_smoke.yaml` files are quick sanity checks.
- `tests/`: smoke and unit tests for configs, builders, replay, trainers, Pommerman integration, analysis, and visualization helpers.
- `analyze_results.py`: regenerates `results/final_results.png`, `results/final_summary.md`, and `results/final_summary.csv` from eval logs.
- `visualize_game.py`, `visualize_imagination.py`, `compare_strategies.py`, `diagnose_policy_behavior.py`: qualitative analysis and debugging tools.
- `scripts/`: setup, validation, final-matrix, poster, and Oscar submission scripts.
- `results/`: committed submission outputs, including final plots, summary tables, eval-only logs, poster figures, and gameplay frames.
- `deliverables/`: final paper PDF, poster PDF, and gameplay preview GIF.
- `docs/`: supporting runbook, visualization guide, and Pommerman background notes.
- `third_party/pommerman/`: bundled Pommerman runtime used by the environment adapter.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev,analysis]"
```

The bundled Pommerman backend is included under `third_party/pommerman`, so the official Pommerman package does not need to be installed separately.

## Quick Validation

```bash
python3 -m pytest
python3 -m madreamer.cli.train --config configs/ppo_smoke.yaml --steps 64
python3 -m madreamer.cli.train --config configs/shared_smoke.yaml --steps 4
python3 analyze_results.py
```

For the full validation pass:

```bash
./scripts/validate_project.sh
```

## Reproducing Results

The submitted plots and tables are generated from eval-only Oscar logs in `results/eval_logs/oscar/`:

```bash
python3 analyze_results.py
```

To analyze a fresh run, point the script at the run directory that contains `<experiment>/logs/metrics.jsonl`:

```bash
python3 analyze_results.py --log-root artifacts/oscar --output-dir results
```

The main submitted outputs are:

- `results/final_summary.md`
- `results/final_summary.csv`
- `results/final_results.png`
- `results/poster_figures/pommerman_results_summary.png`
- `deliverables/paper.pdf`
- `deliverables/final-poster.pdf`

## Running Experiments

Local smoke runs:

```bash
python3 -m madreamer.cli.train --config configs/ppo_smoke.yaml --steps 64
python3 -m madreamer.cli.train --config configs/shared_smoke.yaml --steps 4
```

Local final matrix:

```bash
./scripts/run_matrix.sh
```

Oscar cluster jobs:

```bash
bash scripts/oscar/setup_env.sh
bash scripts/oscar/submit_all.sh
```

## Reading The Results

The strongest result is a controlled, compute-limited comparison rather than a solved Pommerman agent. PPO is the most reliable short-budget baseline; the Dreamer variants show that multi-agent world modeling is difficult, with shared and opponent-aware models sensitive to horizon, opponent behavior, and compounding prediction error. Treat the results as preliminary evidence about the difficulty of model-based multi-agent RL, not as a universal claim about Dreamer.

For more detail, see `docs/runbook.md` and `docs/visualizations.md`.
