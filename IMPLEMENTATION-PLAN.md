# Implementation Plan

## Goal

Build a research pipeline that can answer the core project question:

Can a Dreamer-style world model still help in a multi-agent environment where other agents are part of the dynamics and may also be learning?

We are **not** going to build full multi-agent Dreamer in one step. The plan is to move in stages so that each layer is testable before adding more complexity.

## High-Level Strategy

We will build the project in this order:

1. Environment and trajectory pipeline
2. PPO baseline
3. Single-agent Dreamer-style prototype against fixed opponents
4. Multi-agent Dreamer sharing variants
5. Evaluation, ablations, and write-up

This order matters. If the environment pipeline or PPO baseline does not work, then results from the model-based methods will not be trustworthy.

## Phase 1: Environment + Data Pipeline

### Goal

Get Pommerman producing clean training data through a stable environment adapter.

### What we build

- `PommermanEnv` adapter behind the shared multi-agent environment interface
- symbolic observation encoding into tensors
- rollout collection into replay storage
- support for storing:
  - observations
  - actions
  - rewards
  - next observations
  - done flags
  - other-agent actions when needed

### Initial scope

To reduce complexity, the first version should use:

- `FFA`
- `full observability`
- `no communication`
- one learning agent against fixed opponents

This is the simplest setup that still preserves the strategic structure of Pommerman.

### Why this phase comes first

The environment is the source of all training data. If the observation encoding, action mapping, or reward handling is wrong, every later experiment will be invalid.

### Definition of done

- We can reset and step Pommerman through a clean adapter.
- Observations are converted into consistent tensor-friendly representations.
- Trajectories are saved correctly.
- We can run scripted or random agents through the pipeline without crashes.

## Phase 2: PPO Baseline

### Goal

Build a real model-free baseline that proves the environment and training loop are functioning.

### What we build

- actual PPO training
- logging for reward and win rate
- checkpoint saving
- evaluation code against fixed baselines

### Why PPO comes before Dreamer

PPO gives us a control group. More importantly, it verifies that:

- the environment wrapper works
- reward signals are usable
- learning is possible at all

If PPO fails completely, then Dreamer failing would not tell us much.

### Special caution from the Pommerman paper

The paper suggests that naive PPO struggles without:

- large batch sizes
- reward shaping

So we should be ready to add:

- light reward shaping
- curriculum or easier opponents
- careful tuning of rollout sizes

### Definition of done

- PPO learns above random play
- PPO can be evaluated consistently
- we understand whether reward shaping is necessary

## Phase 3: Single-Agent Dreamer Prototype

### Goal

Get a Dreamer-style world model working in Pommerman before adding full multi-agent learning complexity.

### What we build

- replace the placeholder world model with a real RSSM-style latent dynamics model
- train the world model on real trajectories
- add losses for:
  - latent dynamics
  - reward prediction
  - continuation / termination prediction
  - optionally observation reconstruction
- add imagined rollouts
- train actor and critic using imagined trajectories

### Important simplification

At this stage, we should still use:

- one learning agent
- fixed opponents

This isolates whether model-based learning works in Pommerman at all.

### Why this matters

If the world model cannot learn even against fixed opponents, then multi-agent self-play is premature.

### Definition of done

- the world model can train stably on real trajectories
- imagined rollouts are produced correctly
- the Dreamer-style agent is competitive with the PPO baseline in the fixed-opponent setting

## Phase 4: Multi-Agent Dreamer Variants

### Goal

Implement the actual research comparison from the proposal.

### Variants

#### 1. Independent

- each agent has its own world model
- each agent has its own actor and critic

#### 2. Shared

- all agents share one world model
- each agent has its own actor and critic

#### 3. Opponent-aware

- each agent has a world model conditioned on other agents’ actions
- each agent still has its own actor and critic

### What we compare

- world model sharing strategy
- imagination horizon length
- cooperative vs competitive settings
- Dreamer variants vs PPO baseline

### Why this phase is later

Multi-agent non-stationarity makes everything harder. We want to introduce that only after the single-agent fixed-opponent version is working.

### Definition of done

- all three variants run in the same training framework
- the only differences are controlled by config
- results can be compared fairly across strategies

## Phase 5: Evaluation and Final Analysis

### Metrics

We will track:

- cumulative reward
- win rate
- sample efficiency

### Visualizations

We should produce:

- reward curves over environment steps
- win rate curves
- shared vs independent vs opponent-aware comparison plots
- imagination horizon ablation plots
- Dreamer vs PPO comparison plots
- qualitative imagined rollouts when possible

### Final analysis questions

- Does model-based learning improve sample efficiency in Pommerman?
- Does a shared world model help or hurt compared with independent models?
- Does conditioning on other agents’ actions improve stability or performance?
- Does shorter imagination horizon work better because opponent prediction errors compound quickly?

## First Practical Task Sequence

This is the immediate order of implementation:

1. Implement `PommermanEnv`
2. Lock the first experiment setting:
   - FFA
   - full observability
   - no communication
   - fixed opponents
3. Build PPO training end-to-end
4. Add evaluation and logging
5. Replace the placeholder world model with RSSM-style dynamics
6. Add imagined rollout training
7. Add the three multi-agent sharing variants
8. Run ablations and analyze results

## What We Are Explicitly Not Doing First

To keep scope under control, the first implementation should **not** include:

- learned communication
- partial observability
- full multi-agent self-play from day one
- every ablation at once
- canonical four-learning-agent setup before fixed-opponent experiments work

These can be added later if the core pipeline is stable.

## Design Constraints From the Pommerman Paper

Based on the paper, we should keep the following in mind:

- Pommerman uses **symbolic observations**, not raw pixels
- a CNN is still reasonable, but it should be applied to structured board tensors
- bomb usage creates a strong local optimum, so agents may learn to never bomb unless training is shaped carefully
- PPO and DQN can struggle out of the box
- communication exists in the benchmark, but it should stay out of scope initially

## Bottom Line

The implementation plan is:

- build the environment and data pipeline first
- prove learning works with PPO
- get one Dreamer-style agent working against fixed opponents
- then scale to the `independent`, `shared`, and `opponent_aware` variants
- then run the ablations and evaluate reward, win rate, and sample efficiency

This gives us the best chance of finishing a working project with interpretable results instead of an over-scoped prototype.
