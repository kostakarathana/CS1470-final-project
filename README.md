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

## Results and Visualization Next Steps

This project is a reinforcement learning experiment, so the important outputs are
not only the trained model files. A future developer should look for three kinds
of evidence:

- quantitative results: reward curves, win-rate curves, and summary tables
- qualitative results: gameplay frames or GIFs that show what agents actually do
- diagnostics: checks that the learned world model is not only training, but
  predicting useful future structure

The current final results are already summarized in:

- `results/final_summary.md`: compact table for the report or poster
- `results/final_summary.csv`: same table in spreadsheet-friendly format
- `results/final_results.png`: reward and win-rate curves
- `artifacts/final/**/logs/metrics.jsonl`: raw training and evaluation logs
- `artifacts/final/**/checkpoints/*.pt`: saved PyTorch model checkpoints

If you are new to the project, start with this sequence:

```bash
# 1. Make sure the repo still runs.
./scripts/validate_project.sh

# 2. Regenerate the quantitative plots and tables.
python3 analyze_results.py

# 3. Generate a short gameplay visualization from the strongest baseline.
python3 visualize_game.py --config configs/final/ppo_ffa.yaml \
  --checkpoint artifacts/final/ppo-ffa/checkpoints/final-ppo-ffa_ppo_latest.pt \
  --output artifacts/poster_gameplay.gif --frames 64

# 4. Generate a Dreamer imagination diagnostic.
python3 visualize_imagination.py --config configs/final/shared_h3_ffa.yaml \
  --checkpoint artifacts/final/shared-h3-ffa/checkpoints/final-shared-h3-ffa_shared_latest.pt \
  --output artifacts/shared_h3_imagination.gif --frames 16
```

Read the outputs as follows:

- `eval_mean_reward`: average evaluation reward; higher is better.
- `eval_win_rate`: fraction of evaluation games won.
- `Best Reward`: the best reward seen at any evaluation point.
- `Final Reward`: the reward at the last evaluation point.
- A temporary win-rate spike with poor final reward means the policy was not
  stable enough to claim reliable learning.

For the final poster, the most useful visual assets are:

- `results/final_results.png` for the main quantitative plot
- `results/final_summary.md` for the comparison table
- `artifacts/poster_gameplay.gif` or extracted frames for qualitative behavior
- `artifacts/shared_h3_imagination.gif` for a world-model imagination diagnostic

## Project Next Steps for a New Developer

This repository compares model-free and model-based reinforcement learning in
Pommerman. In plain language, the model-free baseline learns only from real game
experience, while Dreamer learns an internal world model and practices inside
imagined futures. The current code is functional, but the results should be
treated as a first controlled study rather than a solved benchmark.

Recommended next steps:

1. Finish the course deliverables.
   Add a high-resolution horizontal 4:3 poster JPG and a final report PDF or
   LaTeX source. The code and experiments are present, but the course submission
   is incomplete without the poster and writeup.

2. Run longer experiments.
   The final matrix uses only 2048 environment steps so it can run on a local
   CPU. Pommerman is difficult, and stronger conclusions require longer runs,
   more seeds, and more evaluation episodes.

3. Add direct world-model diagnostics.
   The Dreamer results depend on whether the world model predicts useful
   futures. Add metrics such as next-board reconstruction accuracy, reward
   prediction error, continuation prediction accuracy, and imagined-versus-real
   rollout comparisons.

4. Improve qualitative visualizations.
   Generate side-by-side panels showing Proximal Policy Optimization, shared
   Dreamer, and opponent-aware Dreamer on the same board seeds. Good poster
   figures should show bombs, agent movement, survival failures, and any
   successful eliminations.

5. Simplify before scaling.
   If full four-player Pommerman remains unstable, add a smaller two-agent or
   smaller-board experiment. This can test whether the Dreamer world model works
   when opponent behavior and bomb dynamics are easier to predict.

6. Revisit the reward design.
   Sparse win/loss rewards are hard for short runs. The current shaped reward
   gives credit for useful intermediate events such as destroying wood and
   eliminating opponents. Future work should compare sparse and shaped rewards
   explicitly and explain how reward shaping changes behavior.

7. Compare more carefully against the baseline.
   The baseline is Proximal Policy Optimization, a standard model-free policy
   optimization method. The key research question is not only which method wins,
   but whether Dreamer learns faster per environment step and whether its
   imagined rollouts help or hurt policy learning.

8. Document negative results honestly.
   The current results suggest that Proximal Policy Optimization is more stable
   under the short training budget, while Dreamer variants struggle. This is a
   valid research outcome: multi-agent world modeling is difficult because the
   model must predict game physics and opponent behavior at the same time.
