# Pommerman: A Multi-Agent Playground

## What this paper is about

Pommerman is proposed as a benchmark for multi-agent learning, especially for settings that are not just two-player zero-sum games. The authors position it as a benchmark for problems involving:

- competition and cooperation at the same time
- more than two agents
- planning and long-term reasoning
- opponent and teammate modeling
- explicit communication in some variants

Their argument is that much of the field had converged on environments like Go, Chess, and Poker, while general-sum and N-player settings still lacked a shared benchmark.

## Why Pommerman matters

The paper argues that Pommerman is useful because it combines several properties that most existing benchmarks do not provide together:

- four-player gameplay instead of only 1v1
- both free-for-all and team settings
- optional communication channels
- symbolic, low-dimensional observations instead of raw pixels
- discrete state and action spaces

This makes it easier to study high-level strategic behavior without needing the compute budget required for raw-vision environments.

The authors explicitly frame Pommerman as a possible multi-agent analogue of what Atari was for single-agent RL: a standard environment that can unify different research directions.

## Environment details

### Core game structure

- The board is a symmetric `11 x 11` grid.
- There are `4 agents`, one in each corner.
- In team variants, teammates start on opposite corners.
- The board contains:
  - rigid walls: indestructible, impassable
  - wooden walls: destructible, initially block movement
  - passages: traversable tiles
  - power-ups hidden under about half of the wooden walls

The environment guarantees that agents have an accessible path to each other, but initially those paths are blocked by wooden walls.

### Win conditions

- In team games, the game ends when both players on one team are dead.
- In FFA, the game ends when at most one agent remains alive.
- Ties happen if the max step count is reached or the last agents die on the same turn.
- In competitions, tied games can be rerun, and repeated ties can be broken with collapsing walls.

### Action space

Each turn, an agent chooses one of `6` actions:

1. `Stop`
2. `Up`
3. `Left`
4. `Down`
5. `Right`
6. `Bomb`

In communication variants, the agent also emits a two-word message from a dictionary of size eight each turn.

### Observation space

The paper does **not** use pixel observations. The environment exposes a symbolic observation made up of:

- `Board`: 121 integers for the flattened board
- `Position`: 2 integers
- `Ammo`: 1 integer
- `Blast Strength`: 1 integer
- `Can Kick`: 1 binary integer
- `Teammate`: 1 integer
- `Enemies`: 3 integers
- `Bomb Blast Strength`: per-cell bomb strength information
- `Bomb Life`: per-cell bomb timer information
- `Message`: 2 integers in communicative variants

In partially observable settings, the agent only sees a `5 x 5` region around itself, with unseen cells marked as fog.

### Bomb mechanics

- Agents begin with `1` bomb of ammo.
- Ammo is restored when the placed bomb explodes.
- Initial blast strength is `2`.
- Bombs last `10` time steps before exploding.
- Explosions destroy wooden walls, agents, power-ups, and other bombs in blast range.
- Bombs can trigger chain reactions.

### Power-ups

The three power-ups described are:

- `Extra Bomb`: increases ammo
- `Increase Range`: increases blast strength
- `Can Kick`: allows the agent to kick bombs by moving into them

Bomb-kicking changes the dynamics substantially because kicked bombs then move one tile per time step until blocked.

## What research problems Pommerman is meant to support

The paper highlights Pommerman as a benchmark for:

- multi-agent reinforcement learning
- planning
- opponent modeling
- teammate modeling
- game theory in general-sum settings
- emergent communication
- ad hoc teamwork with unseen teammates

This is important: the benchmark is not just about learning reflexive policies. It is meant to support methods that reason about other agents, future consequences, and coordination.

## Why the benchmark is hard

The paper identifies several sources of difficulty.

### 1. More than two players

With four players, FFA does not reduce to standard two-player zero-sum theory. This makes equilibrium reasoning and training stability harder.

### 2. Cooperative and adversarial incentives coexist

The team settings require coordination, while the FFA setting is competitive. That means the benchmark spans both teammate modeling and opponent modeling.

### 3. Communication can matter

Some variants include a cheap-talk communication channel, which introduces another learning problem on top of control.

### 4. Partial observability

Some variants hide most of the board, which forces agents to act under uncertainty.

### 5. The bomb action creates a nasty local optimum

This is one of the most practically important findings in the paper.

The bomb action is necessary for winning, but it is also strongly correlated with losing early in training because weak agents blow themselves up. That creates a bad local optimum where an agent learns to never place bombs, which keeps it safe in the short term but makes it incapable of winning in the long term.

## Early empirical findings from the paper

The paper reports that vanilla deep RL methods struggled.

- Out-of-the-box DQN and PPO did **not** learn to beat the default `SimpleAgent` without:
  - very large batch sizes
  - shaped rewards

- DAgger was more effective as a bootstrapping method and could produce agents roughly at or above the FFA win rate of a single `SimpleAgent`, around `20%`.

- A hybrid search-based approach performed strongly in the early FFA competition.

- Agents discovered a non-obvious bomb-kicking projectile strategy, which the authors present as evidence that the environment can support interesting emergent tactics.

The big takeaway is that Pommerman is not an easy benchmark for standard RL methods, especially if trained naively.

## Competition setup notes

The paper also describes the competition format, which matters if we want to compare against canonical setups:

- FFA competition used four opposing agents.
- NIPS 2018 competition focused on partially observable team play without communication.
- Agents were submitted as Docker containers.
- Each agent exposed an `act` endpoint.
- Competition-time response budget was `100ms`.

The timeout rule is a competition constraint, not a native property of the environment.

## Most important findings for our project

This paper is directly relevant to our multi-agent Dreamer proposal, but it also reveals a few places where we need to tighten our assumptions.

### 1. Pommerman is a good conceptual fit for our question

Our project asks whether a Dreamer-style world model can still work when other agents are part of the dynamics. The paper explicitly positions Pommerman as an environment that requires:

- planning
- opponent modeling
- teammate modeling
- communication in some variants

That aligns well with a model-based RL project.

### 2. The observation format is symbolic, not visual

This is the biggest correction to our current mental model.

The paper does **not** frame Pommerman as a pixel-input benchmark. It provides structured symbolic observations. That means:

- we do not need a raw-vision pipeline
- we should represent the board as structured channels or feature planes
- a CNN can still be useful, but it should be applied to board tensors, not to rendered game screenshots

So the proposal idea of using a CNN encoder is still defensible, but only if we mean a CNN over symbolic board maps.

### 3. Four-agent dynamics are the default

Pommerman is fundamentally built around four players. Our proposal mentioned possibly restricting experiments to two-agent settings for compute reasons. That may be reasonable for a simplified study, but it would not match the canonical Pommerman setup from the paper.

If we stay with Pommerman, we should decide explicitly whether we want:

- canonical four-player experiments
- a simplified custom subset for prototyping
- a different environment if two-agent training is the real target

### 4. Reward shaping or curriculum is likely necessary

Because the paper reports that naive PPO and DQN struggled, and because bomb usage creates a strong local optimum, we should expect training instability from the start.

That means our project likely needs at least one of:

- shaped rewards
- imitation or behavior cloning warm starts
- curriculum learning
- simpler opponents at the beginning
- shorter-horizon tasks first

Without some training support, it is plausible that our model-based agent will fail for reasons unrelated to the world-model idea itself.

### 5. Simple PPO is not enough as a baseline story

Using PPO as the baseline still makes sense, but the paper suggests we should be careful how we interpret it. If PPO performs badly without shaping or large batches, then a weak PPO result may not tell us much.

A stronger evaluation setup would compare against:

- PPO with the same observation encoding and reward shaping
- the repository `SimpleAgent`
- possibly a search-based baseline if available

### 6. Opponent-aware modeling is especially well motivated here

Because the environment includes:

- moving opponents
- chain reactions from bombs
- teammate dependencies in team variants
- hidden information in partially observable settings

an opponent-aware or teammate-aware world model is not just an optional flourish. It may be necessary for useful imagination rollouts.

### 7. Communication should probably be out of scope for the first pass

The paper includes communication variants, but adding learned communication on top of multi-agent model-based RL is probably too much for the first implementation.

A reasonable scope would be:

- start with non-communicative variants
- then compare FFA vs team settings
- only consider communication later if the core learning system works

## Design guidance for this repo

Based on the paper, the project scaffold should eventually support the following choices explicitly:

- `game mode`: FFA or team
- `observability`: full or partial
- `communication`: on or off
- `opponent setup`: random, fixed baseline, or self-play
- `reward shaping`: on or off
- `input encoding`: symbolic features turned into tensors

The current repo already has the right high-level structure for this, but the final environment adapter and model inputs should be designed around symbolic observations instead of image frames.

## Bottom line

Pommerman is a strong fit for a multi-agent world-model project because it naturally stresses planning, multi-agent interaction, and nontrivial strategy. But the paper also makes clear that it is a difficult benchmark where naive RL struggles, bomb usage creates a major optimization trap, and the observation format is symbolic rather than raw visual.

For our project, the most actionable implications are:

- treat the environment as structured state, not pixels
- expect reward shaping or curriculum to matter
- decide early whether we are targeting canonical four-player Pommerman or a reduced prototype
- keep communication out of the first implementation
- use opponent-aware modeling as a core ablation, not a side experiment
