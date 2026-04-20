# CS1470 Final Project

Runnable research scaffold for multi-agent Dreamer-style reinforcement learning.

The repo currently supports:

- `mock_grid` end-to-end training and evaluation
- bundled real Pommerman environment integration
- PPO baseline training
- Dreamer-lite training for `independent`, `shared`, and `opponent_aware`
- terminal demo rendering for the toy environment
- terminal board rendering for real Pommerman

## Install

```bash
python3 -m pip install -e '.[dev]'
```

## Run

Terminal demo:

```bash
madreamer-demo --config configs/base.yaml --steps 8 --sleep 0.15
```

Short PPO training run:

```bash
madreamer-train --config configs/ppo.yaml --steps 64
```

Short Dreamer-style training run:

```bash
madreamer-train --config configs/shared.yaml --steps 64
```

Evaluation from a saved checkpoint:

```bash
madreamer-eval \
  --config configs/shared.yaml \
  --checkpoint artifacts/madreamer-shared-smoke/best.pt \
  --episodes 4
```

Artifacts are written under `artifacts/<experiment_name>/` and include:

- `best.pt`
- `latest.pt`
- `metrics.json`

## Pommerman

The repo includes a bundled upstream Pommerman source tree under `third_party/pommerman`, and the adapter prefers that copy automatically.

If you want to override that with another checkout, you can still set `POMMERMAN_SOURCE_DIR`.

Real Pommerman PPO training:

```bash
madreamer-train --config configs/pommerman_ppo.yaml --steps 64
```

Real Pommerman Dreamer-style training:

```bash
madreamer-train --config configs/pommerman_shared.yaml --steps 64
```

Real Pommerman terminal demo:

```bash
madreamer-demo --config configs/pommerman_shared.yaml --steps 8 --sleep 0.2
```

Real Pommerman animated GIF:

```bash
madreamer-demo \
  --config configs/pommerman_shared.yaml \
  --checkpoint artifacts/pommerman-shared/best.pt \
  --steps 12 \
  --sleep 0 \
  --gif artifacts/pommerman-shared/demo.gif \
  --fps 5 \
  --open
```

Replay a trained real Pommerman Dreamer checkpoint:

```bash
madreamer-demo \
  --config configs/pommerman_shared.yaml \
  --checkpoint artifacts/pommerman-shared/best.pt \
  --steps 12 \
  --sleep 0.2
```

The adapter installs lightweight compatibility stubs for old `rapidjson`, network, and graphics imports so the upstream environment can run headless on modern Python.

## Project Structure

- `src/madreamer/envs/`: environment adapters
- `src/madreamer/models/`: policy and world-model modules
- `src/madreamer/trainers/`: PPO and Dreamer-lite trainers
- `src/madreamer/cli/`: train, eval, and demo entrypoints
- `configs/`: experiment presets
- `tests/`: smoke and integration tests

## Status

This is a working research codebase with real PPO and real world-model training on Pommerman. The Dreamer path is still a compact Dreamer-style implementation rather than a line-by-line reproduction of DreamerV3, but it is no longer a toy mock pipeline.
