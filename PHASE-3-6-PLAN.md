# Phase 3–6 Timeline & Deliverables

**As of Apr 22, 5:20 PM.** Experiments are running (expected completion ~6:05 PM).

---

## Phase 3: Analysis & Plotting (Apr 25–26, ~1 day)

### Immediate (Once Experiments Complete)

```bash
# Run once all 7 experiments finish
python3 analyze_results.py
# Output: artifacts/final_results.png
#         stdout: summary table + statistics
```

### Deliverables
- **Main comparison plot:** 2×1 grid (reward curves, win rate curves)
- **Summary table:** Final metrics for each experiment
- **Ablation insight:** Which sharing strategy helps? What about horizon?
- **Qualitative figure (optional):** One example Pommerman board state or rollout visualization

### Check This
- Are Dreamer variants learning better than PPO?
- Does shared model outperform independent?
- Does horizon 3 beat horizon 1/5?
- Honest framing if results are noisy or Pommerman is hard.

---

## Phase 4: Poster (Apr 27–28, ~1 day)

### Format
- **Size:** Horizontal 4:3 (e.g., 1600×1200px or similar)
- **Tool:** PowerPoint / Figma / PDF
- **Output:** Save as JPG to repo root as `final-poster.jpg`

### Layout (5-section poster)
1. **Title & Team** (top center)
   - *Multi-Agent Dreamer for Pommerman*
   - Team members

2. **Motivation & Problem** (top left)
   - Single-agent Dreamer works in Minecraft
   - Multi-agent is hard: other agents are non-stationary
   - Can we still dream in multiplayer?

3. **Method** (top right)
   - Dreamer-lite RSSM (latent dynamics model)
   - Three world-model strategies: independent / shared / opponent-aware
   - PPO baseline for comparison

4. **Results** (bottom left & center)
   - Comparison plot (2 learning curves from analyze_results.py)
   - Summary table
   - Short captions: "PPO learns faster early, but shared model catches up"
   - Horizon ablation: "h=3 balances imagination length with stability"

5. **Limitations & Future Work** (bottom right)
   - Pommerman sparse rewards are hard
   - Non-stationarity of other agents challenges world models
   - Future: longer training, better opponent modeling, team settings
   - How much did this help sample efficiency?

### Typography
- **Title:** ~36pt
- **Section headers:** ~24pt
- **Body text:** ~14pt
- **Captions:** ~12pt
- Keep text minimal; prioritize visuals

---

## Phase 5: Final Writeup (Apr 29–May 2, ~3 days)

### Required Sections (Use LaTeX template or PDF)

#### 1. **Title & Who**
- *Multi-Agent Dreamer for Pommerman: Evaluating World-Model Sharing in Non-Stationary Multiplayer Environments*
- Team: [names]
- Date: May 2026

#### 2. **Introduction** (0.5 pages)
- Background: single-agent Dreamer (DreamerV3) is strong
- Problem: multi-agent breaks stationarity assumption
- Proposal: compare independent, shared, opponent-aware world models
- Question: does imagining ahead help in a non-stationary game?

#### 3. **Literature Review** (0.75 pages)
- Dreamer/DreamerV3 [Hafner et al.]
- Multi-agent RL challenges [Leibo, Palmer et al.]
- Pommerman environment [Resnick et al.]
- World models in multi-agent settings [cited papers]

#### 4. **Methodology** (1.5 pages)
- **Environment:** Pommerman FFA, 11×11 board, 4 agents, max 256 steps
- **Observation encoding:** one-hot board planes (walls, agents, bombs, flames) + scalars
- **Reward shaping:** win/loss ±5, wood ×0.1, powerup ×0.5, elimination ×2
- **Dreamer-lite architecture:**
  - RSSM prior/posterior KL loss
  - Reconstruction loss (predict next observation)
  - Continuation loss (predict episode end)
  - Reward loss (predict shaping rewards)
  - Imagined actor-critic update (sample from latent dynamics, train policy on imagined rewards)
- **Three strategies:**
  1. Independent: each agent has own world model
  2. Shared: all agents use one world model
  3. Opponent-aware: world model conditions on other agents' actions
- **Baseline:** PPO (policy gradient, no world model)
- **Horizon ablation:** h ∈ {1, 3, 5}

#### 5. **Results** (1.5 pages)
- **Main table:** method, final mean reward, final win rate
- **Learning curves:** reward over env steps, win rate over env steps
- **Ablation results:** horizon effect (if h=3 is better, why?)
- **Interpretation:**
  - Did Dreamer help vs PPO? (sample efficiency?)
  - Which sharing strategy works best?
  - Where did world models break?

#### 6. **Challenges** (0.75 pages)
- Non-stationarity makes it hard for world models to stay accurate
- Sparse Pommerman rewards → shaped rewards needed
- Prediction error compounds in imagination → short horizon may be necessary
- Compute: training multiple agents with world models is expensive
- Honest framing: what went wrong? (If anything did.)

#### 7. **Ethics** (0.25 pages)
- No personal data, no sensitive applications
- Benchmark environment (Pommerman)
- No bias concerns

#### 8. **Reflection** (1 page)
- **Did it work?** Relative to base/target goals, where are we?
- **Lessons learned:** What worked, what didn't, why?
- **Pivots:** Did you change direction? How?
- **Future work:** Longer training? Better opponent models? Cooperative settings?
- **Biggest takeaway:** What did you learn about model-based RL in multi-agent settings?

### Overall Tone
- Honest about limitations; don't oversell results.
- Emphasize the **ablation story**: which components matter?
- Frame as a **research scaffold** and initial evaluation, not a final state-of-the-art method.

---

## Phase 6: Submission Cleanup (May 2–3, ~1 day)

### Checklist

- [ ] All experiments completed, metrics in `artifacts/final/*/logs/metrics.jsonl`
- [ ] `analyze_results.py` run, plots saved to `artifacts/`
- [ ] Poster (JPG) in repo root: `final-poster.jpg`
- [ ] Final writeup (PDF) in repo root: `FINAL-WRITEUP.pdf`
- [ ] README updated with "How to Reproduce" section
- [ ] Clean git status:
  - [ ] Remove tracked `__pycache__` binaries (optional, but good practice)
  - [ ] Keep source code, configs, final plots
  - [ ] Do NOT commit large checkpoints unless essential
- [ ] Test fresh install:
  ```bash
  pip install -e '.[dev]'
  pytest
  python3 -m madreamer.cli.train --config configs/ppo_smoke.yaml --steps 8
  ```
- [ ] Share GitHub link with mentor TA

### Repo Structure (Final)
```
CS1470-final-project/
├── final-poster.jpg              ← Add
├── FINAL-WRITEUP.pdf             ← Add
├── FINAL-PROJECT-ROADMAP.md      ✓
├── EXPERIMENT-RUNBOOK.md         ✓
├── README.md                      (update)
├── analyze_results.py            ✓
├── configs/final/*.yaml          ✓
├── artifacts/
│   ├── final_results.png         ← Phase 3
│   └── final/
│       ├── ppo-ffa/logs/metrics.jsonl
│       ├── shared-h3-ffa/logs/metrics.jsonl
│       └── ...                   ← Phase 2
├── src/madreamer/
├── tests/
└── third_party/pommerman/
```

---

## Priority Ladder (If Time Runs Out)

1. ✓ **Implementation & tests** (done)
2. ✓ **Experiment configs** (done)
3. → **Quantitative results** (Phase 2, in progress)
4. → **Plots & analysis table** (Phase 3, next)
5. → **Poster** (Phase 4, 1 day)
6. → **Final writeup** (Phase 5, 3 days)
7. (Skip if needed) Qualitative visualizations, team results, long training

**Key insight:** Don't spend >2 days on results if you have <7 days left to poster + writeup.  
Better to ship with good structure and honest ablations than perfect numbers.

---

## Timing Summary

| Phase | Dates | Work | Status |
|-------|-------|------|--------|
| 1 | Apr 22 | Scope & configs | ✓ Done |
| 2 | Apr 22–23 | Matrix runs (45 min) | 🔄 Running |
| 3 | Apr 25 | Analysis (1 day) | Queued |
| 4 | Apr 27–28 | Poster (1 day) | Pending |
| 5 | Apr 29–May 2 | Writeup (3 days) | Pending |
| 6 | May 2–3 | Cleanup & submit | Final |

**Deadline:** May 3, 11:50 PM (hard stop)
