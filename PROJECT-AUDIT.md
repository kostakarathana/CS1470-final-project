# Project Audit

Audit date: April 25, 2026.

## Branch State

- `main` is the active branch and is ahead of `origin/main` by three local commits.
- `full-implementation` contains an older alternate implementation with demo/render modules and small smoke artifacts, but its core APIs diverge from the current code. It is useful as reference material, not as a branch to merge wholesale.
- The current working tree includes uncommitted source/docs changes plus generated artifacts. Tracked `__pycache__` files have been removed from the working tree and should be committed as deletions.

## What Is Implemented

- Pommerman adapter with bundled `third_party/pommerman` runtime support.
- Symbolic board observation encoder with board planes, bomb planes, position, ammo, blast strength, and kick features.
- Reward presets for sparse and shaped Pommerman-style rewards.
- Pommerman time-limit truncation based on each config's `env.max_steps`.
- PPO baseline with GAE, clipped objective, checkpointing, evaluation, and JSONL logging.
- Dreamer-lite RSSM world model with CNN encoder, recurrent latent dynamics, reward prediction, continuation prediction, board/scalar reconstruction, and imagined actor-critic updates.
- World-model sharing strategies: `independent`, `shared`, and `opponent_aware`.
- Replay buffer with sequence sampling and opponent-action context.
- Final experiment configs for PPO, sharing ablation, horizon ablation, and a team-mode stretch run.
- Result analysis script that filters real evaluation rows, flags incomplete/missing runs, and emits plot, Markdown, and CSV summaries.
- Checkpoint-backed gameplay, strategy-comparison, and imagination visualization scripts.
- One-command validation through `scripts/validate_project.sh`.
- Smoke/integration tests for config loading, builders, replay, PPO, Dreamer updates, mock grid, and Pommerman backend.

## Current Result Status

Existing final logs show completed evaluation rows for:

- PPO FFA.
- Independent Dreamer, horizon 3, FFA.
- Shared Dreamer, horizon 1, FFA.
- Shared Dreamer, horizon 3, FFA.
- Shared Dreamer, horizon 5, FFA.
- Opponent-aware Dreamer, horizon 3, FFA.
- Shared Dreamer, horizon 3, cooperative team mode.

Incomplete or missing:

- None for the configured final matrix.

Run `python3 analyze_results.py` to regenerate:

- `artifacts/final_results.png`
- `artifacts/final_summary.md`
- `artifacts/final_summary.csv`
- `results/final_results.png`
- `results/final_summary.md`
- `results/final_summary.csv`

## Main Technical Risks

- Current final results are weak and mostly zero-win. That is acceptable for the course if the report frames this honestly as a hard multi-agent RL setting with compute limits, but the final poster still needs clear ablations and failure analysis.
- Core horizon ablation is complete for h=1, h=3, and h=5, and the cooperative team-mode run is complete.
- Poster and final report are not present in the repo.
- The repo currently tracks generated Python bytecode, which makes the code submission look messy.
- Visualizations are now functional with checkpoints, but the imagined board visualization is qualitative and one-step; it should be described as a diagnostic, not proof of long-horizon predictive accuracy.

## Highest-Priority Next Commands

```bash
pytest
python3 analyze_results.py
python3 visualize_game.py --config configs/final/ppo_ffa.yaml \
  --checkpoint artifacts/final/ppo-ffa/checkpoints/final-ppo-ffa_ppo_latest.pt \
  --output artifacts/poster_gameplay.gif --frames 64
python3 compare_strategies.py --configs ppo_ffa shared_h3_ffa opponent_aware_h3_ffa \
  --checkpoints artifacts/final/ppo-ffa/checkpoints/final-ppo-ffa_ppo_latest.pt \
  artifacts/final/shared-h3-ffa/checkpoints/final-shared-h3-ffa_shared_latest.pt \
  artifacts/final/opponent-aware-h3-ffa/checkpoints/final-opponent-aware-h3-ffa_opponent_aware_latest.pt \
  --titles PPO Shared Opponent-Aware --output artifacts/strategy_comparison.png
./scripts/validate_project.sh
```

## Final Report Story

The strongest honest story is:

> We implemented a compact Multi-Agent Dreamer framework and used Pommerman to test whether sharing or opponent-conditioning a learned dynamics model helps under non-stationarity. PPO remained the strongest short-budget baseline in the current runs, while Dreamer variants struggled with sparse wins and compounding prediction error. The negative result is still informative because the controlled ablations show where Dreamer-style imagination breaks first in this multi-agent setting.
