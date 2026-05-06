"""
run_inverse.py

For a fixed set of target epsilon values, find the required sigma for a
chosen variant using the forward-sweep + interpolation approach.

Usage:
    python run_inverse.py --variant variant1
    python run_inverse.py --variant variant2
    python run_inverse.py --variant baseline
    python run_inverse.py --variant single_run

Output:
    - Printed table of (epsilon, sigma) pairs
    - CSV:  results_<variant>.csv
    - Plot: results_<variant>.png
"""

import argparse
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from dp_epsilon_calc import (
    compute_final_epsilon,
    dpsgd_rdp_curve,
    min_eps_over_curve,
)

# ── Fixed epsilon grid ─────────────────────────────────────────────────────
TARGET_EPSILONS = [
    0.450, 0.6, 0.8, 1.0, 1.2, 1.4,
    1.6,   1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0,
]

VARIANT_CHOICES = ["variant1", "variant2", "baseline", "single_run"]

# ── Experiment parameters (MNIST, Koskela & Kulkarni 2023, Table 1) ────────
GAMMA  = 0.0213
T      = 40
Q      = 0.1
MU     = 15
DELTA  = 1e-5
ALPHAS = list(range(2, 65))


def forward_sweep(n_points=120):
    """Sweep sigma -> epsilon for all variants. Returns dict of lists."""
    sigmas = np.logspace(np.log10(0.3), np.log10(100.0), n_points)
    fwd = {v: [] for v in VARIANT_CHOICES}
    fwd["sigma"] = []

    print(f"Running forward sweep ({n_points} sigma values) ...")
    print(f"  T={T}, gamma={GAMMA}, q={Q}, mu={MU}, delta={DELTA}")

    for sigma in sigmas:
        base_curve = dpsgd_rdp_curve(ALPHAS, sigma, GAMMA, T)
        e_single   = min_eps_over_curve(base_curve, DELTA)
        eb, e1, e2 = compute_final_epsilon(
            sigma, GAMMA, T, Q, MU, DELTA, ALPHAS
        )
        fwd["sigma"].append(sigma)
        fwd["single_run"].append(e_single)
        fwd["variant1"].append(e1)
        fwd["variant2"].append(e2)
        fwd["baseline"].append(eb)

    return fwd


def invert(fwd, variant, target_epsilons):
    """
    Interpolate sigma for each target epsilon.
    eps(sigma) is monotone decreasing -> reverse arrays so eps is increasing,
    then use np.interp. Returns (sigma_array, floor_epsilon).
    """
    sigmas_arr = np.array(fwd["sigma"])
    eps_arr    = np.array(fwd[variant])

    eps_rev   = eps_arr[::-1]     # increasing
    sigma_rev = sigmas_arr[::-1]  # corresponding sigmas

    target_arr   = np.array(target_epsilons, dtype=float)
    sigma_interp = np.interp(target_arr, eps_rev, sigma_rev,
                             left=np.nan, right=np.nan)

    floor = float(eps_rev[0])
    sigma_interp[target_arr < floor] = np.nan

    return sigma_interp, floor


def save_csv(variant, target_epsilons, sigmas):
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"results_{variant}.csv"
    
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epsilon", "sigma"])
        for eps, sig in zip(target_epsilons, sigmas):
            writer.writerow([
                f"{eps:.3f}",
                f"{sig:.6f}" if not np.isnan(sig) else "N/A",
            ])
    print(f"CSV  saved -> {path}")
    return path



def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute the required DP-SGD noise multiplier sigma "
            "for each target epsilon, for a chosen tuning variant."
        )
    )
    parser.add_argument(
        "--variant",
        choices=VARIANT_CHOICES,
        required=True,
        help=(
            "Tuning variant to evaluate:\n"
            "  variant1   - Theorem 6 tailored bound "
                           "(tune on X1, train on X\\X1)\n"
            "  variant2   - Subsampling amplification + composition\n"
            "  baseline   - Papernot & Steinke 2022 (no subsampling)\n"
            "  single_run - DP-SGD only, no tuning overhead"
        ),
    )
    parser.add_argument(
        "--n_sweep",
        type=int,
        default=120,
        help="Number of sigma points in the forward sweep (default: 120).",
    )
    args = parser.parse_args()

    fwd = forward_sweep(n_points=args.n_sweep)

    sigmas, floor = invert(fwd, args.variant, TARGET_EPSILONS)

    # ── Print table ────────────────────────────────────────────────────────
    print(f"\nVariant : {args.variant}")
    print(f"Floor   : {floor:.4f}  (minimum achievable epsilon)")
    print(f"\n  {'epsilon':>9} | {'sigma':>10}")
    print("  " + "-" * 24)
    for eps, sig in zip(TARGET_EPSILONS, sigmas):
        sig_str = f"{sig:10.4f}" if not np.isnan(sig) else "       N/A"
        flag    = "  <- below floor" if np.isnan(sig) else ""
        print(f"  {eps:9.3f} | {sig_str}{flag}")

    # ── Save outputs ───────────────────────────────────────────────────────
    save_csv(args.variant,  TARGET_EPSILONS, sigmas)


if __name__ == "__main__":
    main()
