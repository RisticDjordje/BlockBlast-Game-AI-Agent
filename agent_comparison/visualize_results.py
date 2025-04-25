import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CSV_PATH = os.path.join(RESULTS_DIR, "results.csv")

# Read the results CSV
df = pd.read_csv(CSV_PATH)

# Ensure output folder exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Metrics for separate plots
metrics = [
    ("score", "Average Score"),
    ("reward", "Average Reward"),
]

# Compute group means
grouped = df.groupby("agent").mean()

# Bar charts for average score and reward with numeric labels
for metric, title in metrics:
    plt.figure()
    bars = plt.bar(grouped.index, grouped[metric])
    for bar in bars:
        y = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, y, f"{y:.1f}", ha="center", va="bottom"
        )
    plt.title(title)
    plt.xlabel("Agent")
    plt.ylabel(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f"average_{metric}.png")
    plt.savefig(fname)
    print(f"Saved {fname}")

# Combined bar chart for steps, valid_moves, invalid_attempts
combined_metrics = ["steps", "valid_moves", "invalid_attempts"]
combined_means = grouped[combined_metrics]

plt.figure()
indices = np.arange(combined_means.shape[0])
bar_width = 0.8 / len(combined_metrics)
for i, m in enumerate(combined_metrics):
    positions = indices - 0.4 + i * bar_width + bar_width / 2
    plt.bar(positions, combined_means[m], bar_width, label=m)
    for x, y in zip(positions, combined_means[m]):
        plt.text(x, y, f"{y:.1f}", ha="center", va="bottom")

plt.title("Average Steps / Valid / Invalid Movements")
plt.xlabel("Agent")
plt.ylabel("Count")
plt.xticks(indices, combined_means.index, rotation=45)
plt.legend()
plt.tight_layout()
combined_fname = os.path.join(RESULTS_DIR, "combined_moves.png")
plt.savefig(combined_fname)
print(f"Saved {combined_fname}")

# Detailed comparison: Random vs Masked PPO
compare_agents = ["Random", "Masked PPO"]
subset_grouped = df[df["agent"].isin(compare_agents)].groupby("agent").mean()
compare_metrics = [
    ("score", "Average Score"),
    ("reward", "Average Reward"),
    ("steps", "Average Steps"),
    ("valid_moves", "Average Valid Moves"),
    ("invalid_attempts", "Average Invalid Attempts"),
]
for metric, title in compare_metrics:
    plt.figure()
    bars = plt.bar(subset_grouped.index, subset_grouped[metric])
    for bar in bars:
        y = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, y, f"{y:.1f}", ha="center", va="bottom"
        )
    plt.title(f"{title} (Random vs Masked PPO)")
    plt.xlabel("Agent")
    plt.ylabel(title)
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f"compare_{metric}.png")
    plt.savefig(fname)
    print(f"Saved {fname}")

# Histogram of score distributions for Random vs Masked PPO
plt.figure()
for agent in compare_agents:
    scores = df[df["agent"] == agent]["score"]
    plt.hist(scores, bins=20, alpha=0.5, label=agent)
plt.title("Score Distribution: Random vs Masked PPO")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
hist_fname = os.path.join(RESULTS_DIR, "hist_scores_random_vs_masked.png")
plt.savefig(hist_fname)
print(f"Saved {hist_fname}")

# Line plots over episodes for each metric
all_line_metrics = [("score", "Score"), ("reward", "Reward")] + [
    (m, m.replace("_", " ").title()) for m in combined_metrics
]
for metric_label, title in all_line_metrics:
    plt.figure()
    for agent in df["agent"].unique():
        agent_df = df[df["agent"] == agent]
        plt.plot(agent_df["episode"], agent_df[metric_label], label=agent)
    plt.title(f"{title} per Episode")
    plt.xlabel("Episode")
    plt.ylabel(title)
    plt.legend()
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f"per_episode_{metric_label}.png")
    plt.savefig(fname)
    print(f"Saved {fname}")
