# Experiment Runbook

Current phase: scope lock and runnable experiment setup.

## Local Backend Status

On April 22, 2026, the project can run Pommerman through the bundled source tree under `third_party/pommerman`.

That means:

- `configs/ppo_smoke.yaml` works because it uses `mock_grid`.
- Pommerman configs under `configs/final/` are now runnable through `madreamer.envs.PommermanEnv`.
- `scripts/bootstrap_runtime.sh` installs the project and verifies the bundled Pommerman backend.
- Direct `import pommerman` may still fail unless `third_party` is on `PYTHONPATH`; training commands do not require that because the adapter sets the path and compatibility stubs before import.

## Sanity Checks

Run these before long experiments:

```bash
pytest
python3 -m madreamer.cli.train --config configs/ppo_smoke.yaml --steps 24
```

Expected output:

- Tests pass.
- PPO smoke writes a checkpoint under `artifacts/checkpoints/`.

## Final Experiment Matrix

All final configs write isolated metrics and checkpoints under `artifacts/final/<experiment>/`.

### Baseline

```bash
python3 -m madreamer.cli.train --config configs/final/ppo_ffa.yaml
```

### World-Model Sharing Ablation

```bash
python3 -m madreamer.cli.train --config configs/final/independent_h3_ffa.yaml
python3 -m madreamer.cli.train --config configs/final/shared_h3_ffa.yaml
python3 -m madreamer.cli.train --config configs/final/opponent_aware_h3_ffa.yaml
```

### Imagination-Horizon Ablation

```bash
python3 -m madreamer.cli.train --config configs/final/shared_h1_ffa.yaml
python3 -m madreamer.cli.train --config configs/final/shared_h3_ffa.yaml
python3 -m madreamer.cli.train --config configs/final/shared_h5_ffa.yaml
```

### Stretch: Cooperative Team Setting

```bash
python3 -m madreamer.cli.train --config configs/final/team_shared_h3.yaml
```

## Output Paths

| Config | Metrics | Checkpoints |
| --- | --- | --- |
| `configs/final/ppo_ffa.yaml` | `artifacts/final/ppo-ffa/logs/metrics.jsonl` | `artifacts/final/ppo-ffa/checkpoints/` |
| `configs/final/independent_h3_ffa.yaml` | `artifacts/final/independent-h3-ffa/logs/metrics.jsonl` | `artifacts/final/independent-h3-ffa/checkpoints/` |
| `configs/final/shared_h1_ffa.yaml` | `artifacts/final/shared-h1-ffa/logs/metrics.jsonl` | `artifacts/final/shared-h1-ffa/checkpoints/` |
| `configs/final/shared_h3_ffa.yaml` | `artifacts/final/shared-h3-ffa/logs/metrics.jsonl` | `artifacts/final/shared-h3-ffa/checkpoints/` |
| `configs/final/shared_h5_ffa.yaml` | `artifacts/final/shared-h5-ffa/logs/metrics.jsonl` | `artifacts/final/shared-h5-ffa/checkpoints/` |
| `configs/final/opponent_aware_h3_ffa.yaml` | `artifacts/final/opponent-aware-h3-ffa/logs/metrics.jsonl` | `artifacts/final/opponent-aware-h3-ffa/checkpoints/` |
| `configs/final/team_shared_h3.yaml` | `artifacts/final/team-shared-h3/logs/metrics.jsonl` | `artifacts/final/team-shared-h3/checkpoints/` |

## Metrics To Extract

From each `metrics.jsonl`, collect:

- `eval_mean_reward`.
- `eval_win_rate`.
- `env_steps`.
- `world_model_loss` when present.
- `actor_loss` when present.
- `critic_loss` when present.

Poster/report tables should use the final evaluation row from each run. Plots should use every evaluation row over `env_steps`.

## Fallback Plan

If Pommerman cannot be made runnable quickly:

1. Keep the Pommerman adapter and fake-backend tests as implementation evidence.
2. Run all smoke-level checks.
3. Add a small custom/mock-grid matrix to produce the ablation plots.
4. State clearly in the report that official Pommerman runtime installation was the blocker, and that the research framework was validated with fake-backend and mock-grid experiments.

This fallback is weaker than the target result, so use it only if the backend issue threatens the final deliverable timeline.
