# Visualization Guide

The visualization tools render symbolic Pommerman states for qualitative analysis and poster figures.

## Tools

| Script | Use |
| --- | --- |
| `visualize_game.py` | Render real gameplay as a GIF plus selected PNG frames. |
| `visualize_imagination.py` | Compare a real board state with a Dreamer world-model prediction. |
| `compare_strategies.py` | Build a side-by-side grid comparing several trained strategies. |
| `diagnose_policy_behavior.py` | Inspect action distributions and policy behavior from a checkpoint. |

## Real Gameplay

```bash
python3 visualize_game.py \
  --config configs/final/ppo_ffa.yaml \
  --checkpoint artifacts/final/ppo-ffa/checkpoints/final-ppo-ffa_ppo_latest.pt \
  --frames 128 \
  --episodes 1 \
  --fps 10 \
  --output artifacts/poster_gameplay.gif
```

If no checkpoint is provided, the script still renders a game using the configured fallback controllers.

## World-Model Imagination

```bash
python3 visualize_imagination.py \
  --config configs/final/shared_h3_ffa.yaml \
  --checkpoint artifacts/final/shared-h3-ffa/checkpoints/final-shared-h3-ffa_shared_latest.pt \
  --frames 16 \
  --fps 8 \
  --output artifacts/shared_h3_imagination.gif
```

Use this as a qualitative diagnostic, not as proof of long-horizon predictive accuracy.

## Strategy Comparison

```bash
python3 compare_strategies.py \
  --configs ppo_ffa shared_h3_ffa opponent_aware_h3_ffa \
  --checkpoints \
    artifacts/final/ppo-ffa/checkpoints/final-ppo-ffa_ppo_latest.pt \
    artifacts/final/shared-h3-ffa/checkpoints/final-shared-h3-ffa_shared_latest.pt \
    artifacts/final/opponent-aware-h3-ffa/checkpoints/final-opponent-aware-h3-ffa_opponent_aware_latest.pt \
  --titles PPO Shared Opponent-Aware \
  --output artifacts/strategy_comparison.png
```

## Submitted Visual Assets

- `deliverables/gameplay-preview.gif`: compact gameplay preview.
- `results/gameplay_frames/`: representative gameplay frames.
- `results/poster_figures/pommerman_results_summary.png`: poster-ready quantitative figure.
- `deliverables/final-poster.pdf`: final poster deliverable.

## Rendering Legend

| Element | Rendering |
| --- | --- |
| Passage | light gray cell |
| Rigid wall | dark gray cell |
| Wood | brown cell |
| Bomb | red cell with life and range text |
| Flame | orange cell |
| Powerups | green, blue, or magenta cells |
| Agents | colored circles with agent id and ammo count |

The rendering is intentionally symbolic rather than pixel-based so the board, bombs, agents, and powerups are legible in static figures.
