# Scaffolding Plan

This repo should move in four stages.

## Stage 1: Stable Skeleton

- keep a single environment adapter interface
- keep configs as the single source of truth for experiment variants
- make `independent`, `shared`, `opponent_aware`, and `ppo` selectable by config only
- ensure smoke tests run on a tiny environment in seconds

## Stage 2: Environment Integration

- add `PommermanEnv` behind `madreamer.envs.base.MultiAgentEnv`
- normalize observations into channel-first tensors
- define a consistent discrete action vocabulary across adapters

## Stage 3: Real Learning

- replace placeholder world model with Dreamer-style RSSM components
- add reconstruction, reward, continuation, and policy/value losses
- add imagination rollouts with configurable horizon

## Stage 4: Evaluation

- compare cooperative vs competitive modes
- benchmark against PPO with matched compute budgets
- log reward, win rate, and sample efficiency
- save imagined rollout visualizations for qualitative analysis
