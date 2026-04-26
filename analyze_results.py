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

import matplotlib.pyplot as plt


def curve_auc(xs: list[int], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    area = 0.0
    for index in range(1, len(xs)):
        width = float(xs[index] - xs[index - 1])
        area += width * (ys[index] + ys[index - 1]) / 2.0
    return area


def load_metrics(log_dir: Path) -> tuple[list[int], list[float], list[float], int | None, int]:
    """Load evaluation metrics from metrics.jsonl.

    Training rows intentionally do not carry eval metrics. Treating those rows as
    zero-valued evaluations makes interrupted runs look complete, so this helper
    only returns rows with phase == "eval" and explicit eval fields.
    """
    env_steps: list[int] = []
    mean_rewards: list[float] = []
    win_rates: list[float] = []
    last_logged_step: int | None = None
    total_rows = 0

    metrics_file = log_dir / "metrics.jsonl"
    if not metrics_file.exists():
        print(f"Warning: {metrics_file} not found")
        return [], [], [], None, 0

    with open(metrics_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total_rows += 1
                data = json.loads(line)
                if "env_steps" in data:
                    last_logged_step = int(data["env_steps"])
                if (
                    data.get("phase") == "eval"
                    and "eval_mean_reward" in data
                    and "eval_win_rate" in data
                    and "env_steps" in data
                ):
                    env_steps.append(int(data["env_steps"]))
                    mean_rewards.append(float(data["eval_mean_reward"]))
                    win_rates.append(float(data["eval_win_rate"]))

    return env_steps, mean_rewards, win_rates, last_logged_step, total_rows


def main():
    artifacts_dir = Path("artifacts/final")
    output_dirs = [Path("artifacts"), Path("results")]

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
        env_steps, rewards, win_rates, last_logged_step, total_rows = load_metrics(log_dir)
        final_reward = rewards[-1] if rewards else None
        final_win_rate = win_rates[-1] if win_rates else None
        best_reward = max(rewards) if rewards else None
        best_win_rate = max(win_rates) if win_rates else None
        reward_auc = curve_auc(env_steps, rewards) if rewards else None
        win_rate_auc = curve_auc(env_steps, win_rates) if win_rates else None
        all_data[name] = {
            "env_steps": env_steps,
            "rewards": rewards,
            "win_rates": win_rates,
            "last_logged_step": last_logged_step,
            "total_rows": total_rows,
            "final_reward": final_reward,
            "final_win_rate": final_win_rate,
            "best_reward": best_reward,
            "best_win_rate": best_win_rate,
            "reward_auc": reward_auc,
            "win_rate_auc": win_rate_auc,
        }
        print(f"\n{name}:")
        if rewards:
            print(f"  Final Mean Reward: {rewards[-1]:.4f}")
            print(f"  Final Win Rate: {win_rates[-1]:.4f}")
            print(f"  Max Mean Reward: {max(rewards):.4f}")
            print(f"  Evaluation Points: {len(rewards)}")
        elif total_rows:
            print(f"  (No evaluation rows; last logged step: {last_logged_step})")
        else:
            print("  (No data collected yet)")

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Reward curves
    ax = axes[0]
    for name, data in all_data.items():
        if data["rewards"]:
            ax.plot(data["env_steps"], data["rewards"], label=name, marker="o", alpha=0.7)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Mean Reward")
    ax.set_title("Learning Curves: Mean Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Win rate curves
    ax = axes[1]
    for name, data in all_data.items():
        if data["win_rates"]:
            ax.plot(data["env_steps"], data["win_rates"], label=name, marker="s", alpha=0.7)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Win Rate")
    ax.set_title("Learning Curves: Win Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / "final_results.png"
        plt.savefig(plot_path, dpi=150)
        print(f"\n✓ Saved {plot_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Experiment':<30} {'Final Reward':<15} {'Final Win Rate':<15} {'Eval Points':<12}")
    print("-" * 70)
    summary_lines = [
        "| Experiment | Final Reward | Best Reward | Final Win Rate | Best Win Rate | Reward AUC | Eval Points | Last Logged Step |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    csv_lines = [
        "experiment,final_reward,best_reward,final_win_rate,best_win_rate,reward_auc,win_rate_auc,eval_points,last_logged_step,status"
    ]
    for name, data in all_data.items():
        if data["rewards"]:
            reward = data["final_reward"]
            best_reward = data["best_reward"]
            win_rate = data["final_win_rate"]
            best_win_rate = data["best_win_rate"]
            reward_auc = data["reward_auc"]
            win_rate_auc = data["win_rate_auc"]
            eval_points = len(data["rewards"])
            print(f"{name:<30} {reward:>14.4f} {win_rate:>14.4f} {eval_points:>11}")
            summary_lines.append(
                f"| {name} | {reward:.4f} | {best_reward:.4f} | {win_rate:.4f} | "
                f"{best_win_rate:.4f} | {reward_auc:.4f} | {eval_points} | {data['last_logged_step'] or 0} |"
            )
            csv_lines.append(
                f"{name},{reward:.6f},{best_reward:.6f},{win_rate:.6f},{best_win_rate:.6f},"
                f"{reward_auc:.6f},{win_rate_auc:.6f},{eval_points},{data['last_logged_step'] or 0},complete"
            )
        elif data["total_rows"]:
            summary_lines.append(
                f"| {name} | incomplete | incomplete | incomplete | incomplete | incomplete | 0 | "
                f"{data['last_logged_step'] or 0} |"
            )
            csv_lines.append(f"{name},,,,,,,0,{data['last_logged_step'] or 0},incomplete")
        else:
            summary_lines.append(f"| {name} | missing | missing | missing | missing | missing | 0 | 0 |")
            csv_lines.append(f"{name},,,,,,,0,0,missing")

    print("=" * 70)
    for output_dir in output_dirs:
        summary_path = output_dir / "final_summary.md"
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        print(f"✓ Saved {summary_path}")
        csv_path = output_dir / "final_summary.csv"
        csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
        print(f"✓ Saved {csv_path}")


if __name__ == "__main__":
    main()
