# Game Visualization Guide

This project includes interactive visualizations of Pommerman gameplay to illustrate:
1. **Real gameplay** with random agents
2. **Trained agent gameplay** (once models are trained)
3. **Imagination visualizations** showing world model predictions

## Quick Start

### 1. Generate Sample Gameplay (No Training Required)

```bash
# Quick demo (32 frames)
python3 visualize_game.py \
  --config configs/final/ppo_ffa.yaml \
  --frames 32 \
  --output artifacts/quick_demo.gif

# Extended gameplay (128 frames, good for posters)
python3 visualize_game.py \
  --config configs/final/ppo_ffa.yaml \
  --frames 128 \
  --output artifacts/gameplay_full.gif \
  --fps 10
```

**Output:**
- `artifacts/gameplay_full.gif` — animated GIF of 128-frame game
- `artifacts/frame_0000.png`, `frame_0007.png`, etc. — key frame snapshots for poster

### 2. Generate Imagined Rollouts (Requires Trained Model)

Once a model is trained:

```bash
python3 visualize_imagination.py \
  --config configs/final/shared_h3_ffa.yaml \
  --checkpoint artifacts/final/shared-h3-ffa/checkpoints/final-shared-h3-ffa_shared_latest.pt \
  --output artifacts/imagination_demo.gif
```

**Output:**
- GIF showing real board state (left) vs. imagined next states (right)
- Illustrates how well the world model predicts agent movements and explosions

## Visualization Legend

### Board Cells
- **Light Gray** (200,200,200) — Passage (walkable)
- **Dark Gray** (100,100,100) — Rigid wall (indestructible)
- **Brown** (139,90,43) — Wood (destructible)
- **Red** (255,0,0) — Bomb (will explode)
- **Orange** (255,165,0) — Flames (active explosion)
- **Green** (0,255,0) — Powerup: Extra bomb
- **Blue** (0,0,255) — Powerup: Extra range
- **Magenta** (255,0,255) — Powerup: Kick ability

### Agents
Each agent is a colored circle with:
- **Agent number** (0, 1, 2, 3) in the center
- **Ammo count** (a0, a1, etc.) below
- **Colored by player:**
  - Yellow (255,255,0) — Agent 0
  - Orange (255,127,0) — Agent 1
  - Cyan (0,255,255) — Agent 2
  - Pink (255,192,203) — Agent 3

### Metadata
- **Frame number** at bottom left
- **Title** (Real/Imagined) for comparison frames

## Script Reference

### visualize_game.py

Generate visualizations of actual gameplay with random or trained agents.

**Arguments:**
- `--config` (Path) — YAML config file [default: `configs/final/ppo_ffa.yaml`]
- `--checkpoint` (Path) — Optional trained model checkpoint
- `--output` (Path) — Output GIF path [default: `artifacts/gameplay.gif`]
- `--frames` (int) — Max frames per episode [default: 256]
- `--episodes` (int) — Number of episodes to visualize [default: 1]
- `--fps` (int) — GIF frames per second [default: 8]

**Output Files:**
- `.gif` — Animated sequence of all frames
- `frame_XXXX.png` — Key snapshots (initial, 25%, 50%, 75%, end)

### visualize_imagination.py

Generate side-by-side real vs. imagined rollouts from a trained world model.

**Arguments:**
- `--config` (Path) — YAML config file
- `--checkpoint` (Path, required) — Trained model checkpoint
- `--output` (Path) — Output GIF path
- `--frames` (int) — Max frames [default: 64]
- `--fps` (int) — GIF FPS [default: 8]

**Output:**
- GIF with real board state (left) and imagined next state (right)
- Allows visual inspection of world model accuracy

## For Poster Use

### Recommended Visuals

1. **Gameplay sequence** (5-10 key frames showing progression):
   ```bash
   python3 visualize_game.py --frames 128 --output poster_gameplay.gif
   ```
   Use `frame_0000.png`, `frame_0032.png`, `frame_0064.png`, `frame_0096.png`, `frame_0128.png`

2. **Multi-strategy comparison** (if models trained):
   Create gameplay for PPO, shared, and opponent-aware:
   ```bash
   python3 visualize_game.py --config configs/final/ppo_ffa.yaml --output ppo_game.gif
   python3 visualize_game.py --config configs/final/shared_h3_ffa.yaml --output shared_game.gif
   python3 visualize_game.py --config configs/final/opponent_aware_h3_ffa.yaml --output opponent_game.gif
   ```
   Display key frames from each in a grid showing different play styles.

3. **Imagination quality** (qualitative validation):
   ```bash
   python3 visualize_imagination.py --checkpoint <trained_model> --output imagination.gif
   ```
   Show 3-4 frames of real vs. imagined side-by-side to illustrate model accuracy.

## Customization

To modify rendering (colors, size, details), edit these functions:

- `render_board_state()` in `visualize_game.py`
  - Change `cell_size` for larger/smaller board
  - Modify `colors` dict for custom color scheme
  - Add text overlays (rewards, actions, etc.)

- `render_side_by_side()` in `visualize_imagination.py`
  - Add diff visualization (highlight prediction errors)
  - Show attention/saliency maps
  - Display agent observations instead of full board

## Performance Notes

- **Real gameplay** (random agents): ~2 frames/second
- **GIF generation**: ~30 frames/second (PIL encoding)
- For **128-frame GIF**: ~10-15 seconds total runtime
- File sizes: ~50KB per GIF (32 frames), ~100KB (128 frames)

## Troubleshooting

**GIF not created:**
- Check that `--output` parent directory exists
- Script may crash if checkpoint format is wrong

**Rendering is slow:**
- Reduce `--frames` or `--fps` to speed up iteration
- Increase `cell_size` parameter requires code edit

**Imagined rollouts look wrong:**
- World model may not be trained yet; run full experiment first
- Check checkpoint path is correct and model converged

---

**Created:** April 22, 2026
**Status:** Ready for use with trained models (Phases 2-3)
