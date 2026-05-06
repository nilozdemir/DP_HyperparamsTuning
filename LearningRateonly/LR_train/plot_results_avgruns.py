"""
plot_results.py
---------------
Reads tuning_results.csv  (from tuning_script.py)
  and  final_results.csv  (from opacus_mnist.py)
and plots test accuracy vs. epsilon for both stages, matching the style
of Figure 2 in Koskela & Kulkarni (2024).

Usage
-----
    python plot_results.py

Output
------
    accuracy_vs_epsilon.png
"""

import os
import csv

import matplotlib
matplotlib.use("Agg")           # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def read_tuning_csv(path: str) -> tuple[list, list]:
    """Read epsilon and tuning_test_acc from tuning_results.csv."""
    xs, ys = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            xs.append(float(row["epsilon"]))
            ys.append(float(row["tuning_test_acc"]))
    return xs, ys


def read_final_csv(path: str) -> tuple[list, list, list]:
    """Read epsilon, mean_acc, std_acc from final_results.csv (10-run averages)."""
    xs, means, stds = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            xs.append(float(row["epsilon"]))
            means.append(float(row["mean_acc"]))
            stds.append(float(row["std_acc"]))
    return xs, means, stds


# ──────────────────────────────────────────────────────────────────────────────
# Load results
# ──────────────────────────────────────────────────────────────────────────────
TUNING_CSV = "tuning_results.csv"
FINAL_CSV  = "final_results.csv"

missing = [p for p in (TUNING_CSV, FINAL_CSV) if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(
        f"Cannot find: {missing}.\n"
        "Run tuning_script.py then opacus_mnist.py first."
    )

eps_tune, acc_tune               = read_tuning_csv(TUNING_CSV)
eps_final, mean_final, std_final = read_final_csv(FINAL_CSV)

# Convert to percentages
acc_tune_pct  = [a * 100 for a in acc_tune]
mean_final_pct = [a * 100 for a in mean_final]
std_final_pct  = [s * 100 for s in std_final]  # std as percentage points

# Standard error of the mean (n=10) for the error bars, matching the paper
sem_final_pct  = [s / np.sqrt(10) for s in std_final_pct]

# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

# ── Tuning-phase accuracies (single best candidate on subset) ────────────────
ax.plot(
    eps_tune, acc_tune_pct,
    marker="D", markersize=6, linewidth=1.8,
    color="#2196F3", markerfacecolor="#2196F3",
    label="our method — variant 2 (tuning, subset)",
)

# ── Final-training accuracies (mean ± SEM across 10 runs, full dataset) ──────
ax.errorbar(
    eps_final, mean_final_pct,
    yerr=sem_final_pct,
    marker="o", markersize=6, linewidth=1.8, capsize=4, capthick=1.4,
    color="#4CAF50", markerfacecolor="#4CAF50", ecolor="#4CAF50",
    label="our method — variant 2 (final, full dataset, mean ± SEM, n=10)",
)

# ── Formatting ────────────────────────────────────────────────────────────────
ax.set_xlabel("final ε", fontsize=13)
ax.set_ylabel("test accuracy (%)", fontsize=13)
ax.set_title("MNIST, μ=15  (variant 2)", fontsize=13)

ax.set_xlim(left=0.0)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))

ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
ax.legend(fontsize=10, framealpha=0.9)

fig.tight_layout()

OUT = "accuracy_vs_epsilon.png"
fig.savefig(OUT, dpi=150)
print(f"Plot saved → {OUT}")
plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'ε':>6}  {'Tuning acc (%)':>16}  {'Mean final (%)':>16}  {'Std (%)':>9}")
print("-" * 56)

final_map = {e: (m, s) for e, m, s in zip(eps_final, mean_final_pct, std_final_pct)}
for eps, t_acc in zip(eps_tune, acc_tune_pct):
    if eps in final_map:
        m, s = final_map[eps]
        print(f"{eps:>6.3f}  {t_acc:>16.2f}  {m:>16.2f}  {s:>9.2f}")
    else:
        print(f"{eps:>6.3f}  {t_acc:>16.2f}  {'N/A':>16}  {'N/A':>9}")
