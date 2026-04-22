#!/usr/bin/env python3
"""
Analyze final experiment results and generate plots/tables for poster.

Reads metrics.jsonl from each final experiment and produces:
- Comparison tables (sharing strategy, horizon ablation)
- Reward curves (learning over time)
- Win rate curves
- Summary statistics
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(log_dir: Path) -> tuple[List[float], List[float], List[int]]:
    """Load metrics.jsonl and extract env_steps, eval_mean_reward, eval_win_rate."""
    env_steps = []
    mean_rewards = []
    win_rates = []
    
    metrics_file = log_dir / "metrics.jsonl"
    if not metrics_file.exists():
        print(f"Warning: {metrics_file} not found")
        return [], [], []
    
    with open(metrics_file, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                env_steps.append(data.get("env_steps", 0))
                mean_rewards.append(data.get("eval_mean_reward", 0.0))
                win_rates.append(data.get("eval_win_rate", 0.0))
    
    return env_steps, mean_rewards, win_rates


def main():
    artifacts_dir = Path("artifacts/final")
    
    experiments = {
        "PPO": artifacts_dir / "ppo-ffa" / "logs",
        "Independent (h=3)": artifacts_dir / "independent-h3-ffa" / "logs",
        "Shared (h=1)": artifacts_dir / "shared-h1-ffa" / "logs",
        "Shared (h=3)": artifacts_dir / "shared-h3-ffa" / "logs",
        "Shared (h=5)": artifacts_dir / "shared-h5-ffa" / "logs",
        "Opponent-Aware (h=3)": artifacts_dir / "opponent-aware-h3-ffa" / "logs",
        "Team (h=3)": artifacts_dir / "team-shared-h3" / "logs",
    }
    
    print("=" * 70)
    print("FINAL EXPERIMENT ANALYSIS")
    print("=" * 70)
    
    # Load all metrics
    all_data = {}
    for name, log_dir in experiments.items():
        env_steps, rewards, win_rates = load_metrics(log_dir)
        all_data[name] = {
            "env_steps": env_steps,
            "rewards": rewards,
            "win_rates": win_rates,
        }
        print(f"\n{name}:")
        if rewards:
            print(f"  Final Mean Reward: {rewards[-1]:.4f}")
            print(f"  Final Win Rate: {win_rates[-1]:.4f}")
            print(f"  Max Mean Reward: {max(rewards):.4f}")
        else:
            print("  (No data collected yet)")
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Reward curves
    ax = axes[0]
    for name, data in all_data.items():
        if data["rewards"]:
            ax.plot(data["env_steps"], data["rewards"], label=name, marker='o', alpha=0.7)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Mean Reward")
    ax.set_title("Learning Curves: Mean Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Win rate curves
    ax = axes[1]
    for name, data in all_data.items():
        if data["win_rates"]:
            ax.plot(data["env_steps"], data["win_rates"], label=name, marker='s', alpha=0.7)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Win Rate")
    ax.set_title("Learning Curves: Win Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("artifacts/final_results.png", dpi=150)
    print("\n✓ Saved artifacts/final_results.png")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Experiment':<30} {'Final Reward':<15} {'Final Win Rate':<15}")
    print("-" * 70)
    
    for name, data in all_data.items():
        if data["rewards"]:
            reward = data["rewards"][-1]
            wr = data["win_rates"][-1]
            print(f"{name:<30} {reward:>14.4f} {wr:>14.4f}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
