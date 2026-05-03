# Experiment Runbook

This runbook records how to run, validate, and regenerate the final experiment evidence.

## Backend

The project uses the bundled Pommerman source tree in `third_party/pommerman`. The adapter in `src/madreamer/envs/pommerman.py` adds compatibility shims for old Pommerman imports, so normal training commands should work without installing the official playground package.

Useful quick checks:

```bash
python3 -m pytest
python3 -m madreamer.cli.train --config configs/ppo_smoke.yaml --steps 64
python3 -m madreamer.cli.train --config configs/shared_smoke.yaml --steps 4
```

## Final Matrix

Local final configs write metrics and checkpoints under `artifacts/final/<experiment>/`:

| Config | Purpose |
| --- | --- |
| `configs/final/ppo_ffa.yaml` | PPO baseline |
| `configs/final/independent_h3_ffa.yaml` | Dreamer with independent world models |
| `configs/final/shared_h3_ffa.yaml` | Dreamer with a shared world model |
| `configs/final/opponent_aware_h3_ffa.yaml` | Dreamer conditioned on opponent actions |
| `configs/final/shared_h1_ffa.yaml` | Shared Dreamer, horizon 1 |
| `configs/final/shared_h5_ffa.yaml` | Shared Dreamer, horizon 5 |
| `configs/final/team_shared_h3.yaml` | Cooperative team-mode stretch run |

Run the local matrix from the repo root:

```bash
./scripts/run_matrix.sh
```

Oscar configs under `configs/oscar/` use longer budgets and write to `artifacts/oscar/<experiment>/`.

## Regenerating Submitted Outputs

The committed submission uses eval-only logs in `results/eval_logs/oscar/`. This keeps the repository navigable while preserving enough raw evidence to regenerate the final curves and table.

```bash
python3 analyze_results.py
```

To analyze a fresh local or cluster run:

```bash
python3 analyze_results.py --log-root artifacts/oscar --output-dir results
```

The script writes:

- `results/final_results.png`
- `results/final_summary.md`
- `results/final_summary.csv`

## Metrics

- `eval_mean_reward`: mean reward during evaluation episodes.
- `eval_win_rate`: fraction of evaluation games won.
- `Reward AUC`: trapezoidal area under the reward curve over environment steps.
- `Eval Points`: number of evaluation rows found in the metrics log.
- `Last Eval Step`: final evaluation `env_steps` value observed in the log.

Training rows are intentionally ignored by `analyze_results.py`; only rows with `phase == "eval"` and explicit evaluation fields are used.

## Submission Evidence

The final code submission should point reviewers to:

- `README.md` for the project map and commands.
- `results/final_summary.md` for the concise result table.
- `results/final_results.png` for learning curves.
- `deliverables/final-poster.pdf` for the poster.
- `docs/visualizations.md` for qualitative rendering commands.
