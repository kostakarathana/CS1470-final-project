# Final Project Roadmap

Current date: April 22, 2026. Final project deadline: May 3, 2026 at 11:50 PM.

## Current Status

The codebase is mostly at the "implementation complete, experiments beginning" stage.

Built:

- Config-driven PyTorch training pipeline.
- Pommerman adapter with symbolic board feature encoding and bundled runtime support.
- PPO baseline.
- Dreamer-lite trainer with RSSM-style world model, reward prediction, continuation prediction, reconstruction, and imagined actor/critic updates.
- Three world-model sharing strategies: `independent`, `shared`, and `opponent_aware`.
- Replay buffer with sequence sampling and opponent-action context.
- Smoke and integration tests.

Still needed:

- Meaningful experiment runs.
- Reward and win-rate plots.
- Sample-efficiency comparison.
- Imagination-horizon ablation.
- Final poster.
- Final report/reflection.

## Scope Lock

Treat implementation as mostly frozen unless an experiment reveals a blocking bug.

Phase-one execution details are now recorded in `EXPERIMENT-RUNBOOK.md`, including local backend status, exact experiment commands, and output paths.

Final experimental scope:

- PPO baseline.
- Dreamer `shared`.
- Dreamer `independent`.
- Dreamer `opponent_aware`.
- Imagination horizon ablation over at least `1`, `3`, and `5`.
- Pommerman FFA as the main setting.
- Team/cooperative setting only as a stretch goal.

Minimum viable final claim:

> We implemented a compact Multi-Agent Dreamer framework and evaluated whether world-model sharing and opponent conditioning help in a difficult multi-agent symbolic environment. The main contribution is the controlled comparison and analysis, even if absolute performance is limited by Pommerman difficulty and compute.

## Experiment Plan

Run these first:

| Experiment | Purpose |
| --- | --- |
| `ppo_dev` | Model-free PPO baseline |
| `shared_dev` | Main shared-world-model Dreamer result |
| `independent_study` | Separate world models per agent |
| `shared_study` | Shared world model with matched study budget |
| `opponent_aware_study` | Opponent-conditioned dynamics model |

Then run a horizon ablation, preferably using the shared model:

| Variant | Imagination Horizon |
| --- | --- |
| shared-h1 | 1 |
| shared-h3 | 3 |
| shared-h5 | 5 |

Track:

- Mean reward.
- Win rate.
- Environment steps.
- World model loss.
- Actor loss.
- Critic loss.
- Runtime if easy to capture.

## Results To Produce

Needed for poster and report:

- Reward over environment steps.
- Win rate over environment steps.
- Final comparison table: PPO vs shared vs independent vs opponent-aware.
- Horizon ablation table or plot.
- One qualitative figure.

Qualitative figure options:

- Pommerman board trajectory.
- Observation encoding visualization.
- Imagined rollout visualization if there is time to implement it.
- Failure-case trajectory showing unstable policy or bad bombing behavior.

## Poster Plan

Poster format: one high-resolution horizontal 4:3 JPG.

Sections:

- Title: Multi-Agent Dreamer for Pommerman.
- Motivation: Dreamer works well in single-agent settings, but multi-agent environments are non-stationary.
- Method: PPO baseline, Dreamer-lite RSSM, and three world-model sharing strategies.
- Environment: Pommerman symbolic grid observations.
- Results: two plots plus one compact table.
- Discussion: what worked, what failed, and why.
- Limitations and future work: compute, Pommerman difficulty, reward shaping, longer training, stronger opponent modeling.

The poster should be readable in a 2-minute presentation. Do not include every implementation detail.

## Final Report Plan

Required sections:

- Title.
- Who.
- Introduction.
- Literature Review.
- Methodology.
- Results.
- Challenges.
- Ethics.
- Reflection.

Suggested framing:

> This project investigates whether Dreamer-style world models remain useful in multi-agent environments where other agents are part of the dynamics. We built a compact Multi-Agent Dreamer pipeline in PyTorch, evaluated world-model sharing strategies in Pommerman, and compared against PPO under controlled compute.

Important discussion points:

- Pommerman is harder than expected because bombing creates poor local optima.
- Symbolic observations made CNN-style board encoders appropriate, but this is not a raw-pixel benchmark.
- Opponent-aware modeling is conceptually well motivated, but may need more data than the small study budget provides.
- Shorter imagination horizons may be more stable because opponent prediction errors compound quickly.
- Negative or noisy results are acceptable if the ablations are clear and honestly interpreted.

## Timeline

### April 22

- Freeze implementation scope.
- Commit this roadmap.
- Decide exact experiment commands and output paths.

### April 23-25

- Run core experiment matrix.
- Save metrics and checkpoints.
- Start preliminary plots.

### April 25-26

- Finish reward, win-rate, and ablation plots.
- Create final comparison table.
- Decide the main story based on actual results.

### April 27-28

- Build poster draft.
- Keep poster focused on motivation, method, results, and discussion.
- Get team feedback.

### April 29-May 2

- Write final report.
- Adapt introduction/methodology from proposal and implementation notes.
- Add results and challenges honestly.

### May 2-3

- Final repo cleanup.
- Verify install and tests.
- Add poster JPG and final report PDF.
- Ensure README explains how to reproduce the main results.

## Priority Order If Time Gets Tight

1. PPO vs Dreamer quantitative results.
2. Sharing-strategy ablation.
3. Horizon ablation.
4. Poster.
5. Final writeup.
6. Qualitative visualization.
7. Team/cooperative experiments.

The main risk is spending too much time improving the model and not enough time producing results, plots, and writing.
