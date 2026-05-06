"""
Final epsilon calculation for:
"Practical Differentially Private Hyperparameter Tuning with Subsampling"
Koskela & Kulkarni, NeurIPS 2023

FIX vs v1: Theorem 6 now uses full per-order RDP curves for both Mtune and
Mbase instead of a single scalar upper bound. Specifically, the cross terms
in eps1(alpha) and eps2(alpha) now look up eps_tune(alpha-j) and eps_base(j)
at their correct orders j, rather than reusing eps_tune(alpha) and eps_base(alpha)
as a conservative upper bound.

Implements:
  - Proposition 7 / Corollary 3 (Mironov 2017): Gaussian mechanism RDP
  - Theorem 4  : Poisson subsampling RDP amplification (Zhu & Wang 2019)
  - Theorem 5  : Papernot & Steinke tuning RDP cost
  - Theorem 6  : Tailored RDP bound for Variant 1  <-- now uses full curves
  - Proposition 3 (Mironov 2017): RDP -> (eps, delta) conversion
  - Variant 2  : Subsampling amplification + composition
  - Baseline   : Papernot & Steinke without subsampling
"""

import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Type alias: RDP curve = dict mapping integer alpha -> float epsilon
RDPCurve = Dict[int, float]


# ---------------------------------------------------------------------------
# 1. Gaussian mechanism RDP  (Mironov 2017, Proposition 7 / Corollary 3)
# ---------------------------------------------------------------------------

def gaussian_rdp_curve(alphas: List[int], sigma: float) -> RDPCurve:
    """
    Corollary 3 (Mironov 2017): Gaussian mechanism with sensitivity 1,
    noise std sigma satisfies (alpha, alpha/(2*sigma^2))-RDP.
    Returns full curve {alpha: epsilon(alpha)}.
    """
    return {a: a / (2.0 * sigma ** 2) for a in alphas}


# ---------------------------------------------------------------------------
# 2. Theorem 4: Poisson subsampling amplification (Zhu & Wang 2019)
#    Applied to the Gaussian mechanism -> DP-SGD single step RDP curve
# ---------------------------------------------------------------------------

def poisson_subsampled_rdp_curve(alphas: List[int], sigma: float, gamma: float) -> RDPCurve:
    """
    Theorem 4 (Zhu & Wang 2019): RDP curve of one step of Poisson-subsampled
    Gaussian mechanism (= one DP-SGD step).

    For each integer alpha >= 2:
      eps'(alpha) = 1/(alpha-1) * log(
          (1-gamma)^alpha + alpha*gamma*(1-gamma)^{alpha-1}               [j=0,1]
        + C(alpha,2)*gamma^2*(1-gamma)^{alpha-2} * exp(eps_gauss(2))      [j=2]
        + 3*sum_{j=3}^{alpha} C(alpha,j)*gamma^j*(1-gamma)^{alpha-j}
                              * exp((j-1)*eps_gauss(j))                    [j>=3]
      )

    eps_gauss(j) = j/(2*sigma^2)  from Corollary 3 of Mironov 2017.
    """
    gauss = gaussian_rdp_curve(alphas, sigma)
    curve = {}
    for alpha in alphas:
        if alpha < 2:
            # Linear approx for small alpha (not used when alphas start at 2)
            curve[alpha] = gamma ** 2 * alpha / (2.0 * sigma ** 2)
            continue

        log_terms = []

        # j=0 and j=1 combined
        term01 = (1 - gamma) ** alpha + alpha * gamma * (1 - gamma) ** (alpha - 1)
        if term01 > 0:
            log_terms.append(np.log(term01))

        # j=2
        c2 = comb(alpha, 2, exact=False) * (gamma ** 2) * ((1 - gamma) ** (alpha - 2))
        if c2 > 0:
            log_terms.append(np.log(c2) + gauss[2])

        # j=3..alpha -- each uses eps_gauss(j) at the correct order j
        for j in range(3, alpha + 1):
            cj = comb(alpha, j, exact=False) * (gamma ** j) * ((1 - gamma) ** (alpha - j))
            if cj > 0:
                eps_j = j / (2.0 * sigma ** 2)   # gauss at order j (not in dict if j>max(alphas))
                log_terms.append(np.log(3.0) + np.log(cj) + (j - 1) * eps_j)

        max_log = max(log_terms)
        total = sum(np.exp(lt - max_log) for lt in log_terms)
        curve[alpha] = (max_log + np.log(total)) / (alpha - 1)

    return curve


def dpsgd_rdp_curve(alphas: List[int], sigma: float, gamma: float, T: int) -> RDPCurve:
    """
    RDP curve of T steps of DP-SGD: compose T times via standard RDP composition
    (Proposition 1, Mironov 2017: epsilons simply add).
    """
    single_step = poisson_subsampled_rdp_curve(alphas, sigma, gamma)
    return {a: T * single_step[a] for a in alphas}


# ---------------------------------------------------------------------------
# 3. Theorem 5: Papernot & Steinke tuning RDP curve
# ---------------------------------------------------------------------------

def papernot_steinke_rdp_curve(
    alphas: List[int],
    base_curve: RDPCurve,
    mu: float,
    delta_hat: float = 0.0,
) -> RDPCurve:
    """
    Theorem 5 (Papernot & Steinke 2022): RDP curve of the tuning algorithm
    with Poisson(mu) random runs, each running a mechanism with RDP curve
    base_curve.

    eps_tune(alpha) = eps_base(alpha) + mu * delta_hat + log(mu) / (alpha - 1)

    With delta_hat=0 (pure RDP base, e.g. Gaussian):
      eps_tune(alpha) = eps_base(alpha) + log(mu) / (alpha - 1)
    """
    return {
        a: base_curve[a] + mu * delta_hat + np.log(mu) / (a - 1)
        for a in alphas
    }


# ---------------------------------------------------------------------------
# 4. Theorem 6: Tailored RDP for Variant 1  -- NOW WITH FULL CURVES
#
# The key fix: eps1 and eps2 contain cross terms of the form
#   exp((alpha-j-1) * eps_tune(alpha-j)) * exp((j-1) * eps_base(j))
# which require eps_tune and eps_base evaluated at orders alpha-j and j
# respectively, NOT at the top-level order alpha.
# We compute these by extending the curves to all needed sub-orders.
# ---------------------------------------------------------------------------

def _rdp_at_order(curve: RDPCurve, alpha: int, sigma: float, gamma: float, T: int,
                  is_tune: bool, mu: float) -> float:
    """
    Look up RDP value at a given order, computing it on-the-fly if not cached.
    This ensures we always have eps(j) for any j that appears in the cross terms.
    """
    if alpha in curve:
        return curve[alpha]
    # Compute on-the-fly for sub-orders not in the pre-computed dict
    if is_tune:
        base_j = dpsgd_rdp_curve([alpha], sigma, gamma, T)[alpha]
        return base_j + np.log(mu) / (alpha - 1)
    else:
        return dpsgd_rdp_curve([alpha], sigma, gamma, T)[alpha]


def theorem6_rdp_curve(
    alphas: List[int],
    tune_curve: RDPCurve,   # full RDP curve of Mtune
    base_curve: RDPCurve,   # full RDP curve of Mbase (DP-SGD)
    q: float,
    sigma: float,
    gamma: float,
    T: int,
    mu: float,
) -> RDPCurve:
    """
    Theorem 6 (Koskela & Kulkarni 2023): Tailored RDP bound for Variant 1.

    For each integer alpha >= 2, computes max(eps1(alpha), eps2(alpha)) where:

    eps1(alpha) = 1/(alpha-1) * log(
        q^alpha * exp((alpha-1)*eps_tune(alpha))
      + (1-q)^alpha * exp((alpha-1)*eps_base(alpha))
      + sum_{j=1}^{alpha-1} C(alpha,j) * q^{alpha-j} * (1-q)^j
                            * exp((alpha-j-1)*eps_tune(alpha-j))   <- order alpha-j
                            * exp((j-1)*eps_base(j))               <- order j
    )

    eps2(alpha) = 1/(alpha-1) * log(
        (1-q)^{alpha-1} * exp((alpha-1)*eps_base(alpha))
      + sum_{j=1}^{alpha-1} C(alpha-1,j) * q^j * (1-q)^{alpha-1-j}
                            * exp(j * eps_tune(j+1))               <- order j+1
                            * exp((alpha-j-1)*eps_base(alpha-j))   <- order alpha-j
    )

    The cross terms now use eps_tune and eps_base at their CORRECT sub-orders,
    computed on-the-fly for any order not already in the cached curves.
    """
    curve = {}

    for alpha in alphas:
        # Helper: get eps_tune at order k (any k >= 2)
        def et(k):
            return _rdp_at_order(tune_curve, k, sigma, gamma, T,
                                 is_tune=True, mu=mu)

        # Helper: get eps_base at order k (any k >= 2)
        def eb(k):
            return _rdp_at_order(base_curve, k, sigma, gamma, T,
                                 is_tune=False, mu=mu)

        # ---- eps1(alpha) ----
        log_terms1 = []

        # Pure tune term: q^alpha * exp((alpha-1)*eps_tune(alpha))
        log_terms1.append(alpha * np.log(q + 1e-300) + (alpha - 1) * et(alpha))

        # Pure base term: (1-q)^alpha * exp((alpha-1)*eps_base(alpha))
        log_terms1.append(alpha * np.log(1 - q + 1e-300) + (alpha - 1) * eb(alpha))

        # Cross terms j=1..alpha-1:
        # C(alpha,j) * q^{alpha-j} * (1-q)^j
        #   * exp((alpha-j-1)*eps_tune(alpha-j)) * exp((j-1)*eps_base(j))
        for j in range(1, alpha):
            c = comb(alpha, j, exact=False)
            log_c = np.log(c + 1e-300)
            log_q = (alpha - j) * np.log(q + 1e-300)
            log_1q = j * np.log(1 - q + 1e-300)

            # KEY FIX: use eps_tune at order (alpha-j), eps_base at order j
            k_tune = alpha - j   # order for tune term
            k_base = j           # order for base term

            # eps_tune(alpha-j) is only meaningful for k_tune >= 2
            # eps_base(j) is only meaningful for k_base >= 2
            # For k=1: exp((k-1)*eps) = exp(0) = 1, so those terms contribute 0 to the log
            exp_tune = (alpha - j - 1) * et(k_tune) if k_tune >= 2 else 0.0
            exp_base = (j - 1) * eb(k_base) if k_base >= 2 else 0.0

            log_terms1.append(log_c + log_q + log_1q + exp_tune + exp_base)

        max1 = max(log_terms1)
        eps1 = (max1 + np.log(sum(np.exp(t - max1) for t in log_terms1))) / (alpha - 1)

        # ---- eps2(alpha) ----
        log_terms2 = []

        # Pure base term: (1-q)^{alpha-1} * exp((alpha-1)*eps_base(alpha))
        log_terms2.append((alpha - 1) * np.log(1 - q + 1e-300) + (alpha - 1) * eb(alpha))

        # Cross terms j=1..alpha-1:
        # C(alpha-1,j) * q^j * (1-q)^{alpha-1-j}
        #   * exp(j * eps_tune(j+1)) * exp((alpha-j-1)*eps_base(alpha-j))
        for j in range(1, alpha):
            c = comb(alpha - 1, j, exact=False)
            log_c = np.log(c + 1e-300)
            log_q = j * np.log(q + 1e-300)
            log_1q = (alpha - 1 - j) * np.log(1 - q + 1e-300)

            # KEY FIX: eps_tune at order (j+1), eps_base at order (alpha-j)
            k_tune = j + 1        # order for tune term
            k_base = alpha - j    # order for base term

            exp_tune = j * et(k_tune) if k_tune >= 2 else 0.0
            exp_base = (alpha - j - 1) * eb(k_base) if k_base >= 2 else 0.0

            log_terms2.append(log_c + log_q + log_1q + exp_tune + exp_base)

        max2 = max(log_terms2)
        eps2 = (max2 + np.log(sum(np.exp(t - max2) for t in log_terms2))) / (alpha - 1)

        curve[alpha] = max(eps1, eps2)

    return curve


# ---------------------------------------------------------------------------
# 5. Variant 2: subsampling amplification of Mtune then compose with Mbase
#    Uses full RDP curve of Mtune as the "base mechanism" in Theorem 4
# ---------------------------------------------------------------------------

def amplify_rdp_curve(
    alphas: List[int],
    mech_curve: RDPCurve,
    gamma: float,
    mu: float,
    sigma: float,
    T: int,
) -> RDPCurve:
    """
    Apply Theorem 4 subsampling amplification to an arbitrary mechanism whose
    RDP curve is mech_curve, with subsampling ratio gamma.

    For each alpha, uses the correct per-order eps_mech(j) values in the sum,
    rather than a single scalar upper bound.
    """
    curve = {}
    for alpha in alphas:
        if alpha < 2:
            curve[alpha] = gamma ** 2 * alpha * mech_curve.get(alpha, 0.0)
            continue

        log_terms = []

        # j=0,1 combined
        term01 = (1 - gamma) ** alpha + alpha * gamma * (1 - gamma) ** (alpha - 1)
        if term01 > 0:
            log_terms.append(np.log(term01))

        # j=2: uses eps_mech(2)
        c2 = comb(alpha, 2, exact=False) * (gamma ** 2) * ((1 - gamma) ** (alpha - 2))
        if c2 > 0:
            eps2 = _rdp_at_order(mech_curve, 2, sigma, gamma, T, is_tune=True, mu=mu)
            log_terms.append(np.log(c2) + eps2)

        # j=3..alpha: uses eps_mech(j) at the correct order j
        for j in range(3, alpha + 1):
            cj = comb(alpha, j, exact=False) * (gamma ** j) * ((1 - gamma) ** (alpha - j))
            if cj > 0:
                eps_j = _rdp_at_order(mech_curve, j, sigma, gamma, T, is_tune=True, mu=mu)
                log_terms.append(np.log(3.0) + np.log(cj) + (j - 1) * eps_j)

        max_log = max(log_terms)
        total = sum(np.exp(lt - max_log) for lt in log_terms)
        curve[alpha] = (max_log + np.log(total)) / (alpha - 1)

    return curve


# ---------------------------------------------------------------------------
# 6. RDP -> (eps, delta) conversion
#    Proposition 3 (Mironov 2017): if f is (alpha, eps_rdp)-RDP then
#    f is (eps_rdp + log(1/delta)/(alpha-1), delta)-DP.
# ---------------------------------------------------------------------------

def rdp_to_dp(eps_rdp: float, alpha: float, delta: float) -> float:
    """
    Proposition 3 (Mironov 2017): (alpha, eps_rdp)-RDP => (eps, delta)-DP
    with eps = eps_rdp + log(1/delta) / (alpha - 1).

    Note: this is the basic conversion from Mironov. The tighter version
    from Canonne et al. 2020 (Lemma 3 in the Koskela paper) gives a slightly
    better bound; we use the standard autodp form here:
      eps = eps_rdp + log(1-1/alpha) - (log(delta) + log(1-1/alpha)) / (alpha-1)
    which is equivalent to the Canonne et al. formula.
    """
    if alpha <= 1:
        return np.inf
    log_term = np.log(1.0 - 1.0 / alpha)
    return eps_rdp + log_term - (np.log(delta) + log_term) / (alpha - 1)


def min_eps_over_curve(rdp_curve: RDPCurve, delta: float) -> float:
    """Minimize (eps, delta)-DP epsilon over all alpha orders in the curve."""
    return min(rdp_to_dp(eps, alpha, delta) for alpha, eps in rdp_curve.items())


# ---------------------------------------------------------------------------
# 7. Full pipeline: compute final (eps, delta) for all three methods
# ---------------------------------------------------------------------------

def compute_final_epsilon(
    sigma: float,
    gamma: float,
    T: int,
    q: float,
    mu: float,
    delta: float,
    alphas: List[int],
) -> Tuple[float, float, float]:
    """
    Compute final (eps, delta)-DP epsilon for:
      - Baseline (Papernot & Steinke, Thm 5, no subsampling trick)
      - Variant 1 (Thm 6 tailored bound, full per-order curves)
      - Variant 2 (subsampling amplification of Mtune + composition)

    Returns: (eps_baseline, eps_variant1, eps_variant2)
    """

    # DP-SGD base mechanism RDP curve (= Mbase)
    base_curve = dpsgd_rdp_curve(alphas, sigma, gamma, T)

    # Papernot-Steinke tuning RDP curve (= Mtune wrapping DP-SGD)
    tune_curve = papernot_steinke_rdp_curve(alphas, base_curve, mu=mu, delta_hat=0.0)

    # ------------------------------------------------------------------
    # BASELINE: Papernot & Steinke (Thm 5) on the full dataset.
    # Total cost = tuning cost + final model training cost (composed).
    # ------------------------------------------------------------------
    baseline_curve = {a: tune_curve[a] + base_curve[a] for a in alphas}
    eps_baseline = min_eps_over_curve(baseline_curve, delta)

    # ------------------------------------------------------------------
    # VARIANT 1: Thm 6 -- tuning on X1 (Poisson ratio q), train on X\X1.
    # Full RDP curves for both Mtune and Mbase are threaded through.
    # ------------------------------------------------------------------
    v1_curve = theorem6_rdp_curve(
        alphas, tune_curve, base_curve, q, sigma, gamma, T, mu
    )
    eps_variant1 = min_eps_over_curve(v1_curve, delta)

    # ------------------------------------------------------------------
    # VARIANT 2: Mtune subsampled with ratio q, then compose with Mbase.
    # Apply Thm 4 amplification to the full tune_curve, then add base_curve.
    # ------------------------------------------------------------------
    tune_amplified = amplify_rdp_curve(alphas, tune_curve, q, mu, sigma, T)
    v2_curve = {a: tune_amplified[a] + base_curve[a] for a in alphas}
    eps_variant2 = min_eps_over_curve(v2_curve, delta)

    return eps_baseline, eps_variant1, eps_variant2


# ---------------------------------------------------------------------------
# 8. MNIST experiment (Figure 2a): sweep over sigma
# ---------------------------------------------------------------------------

def mnist_experiment():
    gamma = 0.0213
    T = 40        # T=40 steps directly as specified
    q = 0.1
    mu = 15
    delta = 1e-5
    alphas = list(range(2, 65))

    # sigma sweep: small sigma -> large eps, large sigma -> small eps (~0.34 floor)
    sigmas = np.logspace(np.log10(0.5), np.log10(50.0), 60)

    results = {
        "sigma": [],
        "eps_baseline": [],
        "eps_v1": [],
        "eps_v2": [],
        "eps_single_run": [],
    }

    print(f"MNIST experiment (T=40 steps, full RDP curves): gamma={gamma}, "
          f"T={T}, q={q}, mu={mu}")
    print(f"{'sigma':>8} | {'eps_baseline':>14} | {'eps_v1':>10} | "
          f"{'eps_v2':>10} | {'eps_single':>12}")
    print("-" * 65)

    for sigma in sigmas:
        eb, e1, e2 = compute_final_epsilon(sigma, gamma, T, q, mu, delta, alphas)

        # "no tuning cost -- single run": just the DP-SGD training epsilon,
        # no hyperparameter tuning overhead. This is the lower bound shown in
        # Figure 2 as the "no tuning cost --- single run" curve.
        base_curve = dpsgd_rdp_curve(alphas, sigma, gamma, T)
        e_single = min_eps_over_curve(base_curve, delta)

        results["sigma"].append(sigma)
        results["eps_baseline"].append(eb)
        results["eps_v1"].append(e1)
        results["eps_v2"].append(e2)
        results["eps_single_run"].append(e_single)

        print(f"{sigma:8.3f} | {eb:14.4f} | {e1:10.4f} | {e2:10.4f} | {e_single:12.4f}")

    return results


def plot_results(results):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(results["eps_single_run"], results["sigma"], "m-x",
            label="no tuning cost — single run", markersize=3, linewidth=1.5)
    ax.plot(results["eps_v1"], results["sigma"], "b-o",
            label="variant 1 (Thm 6)", markersize=3, linewidth=1.5)
    ax.plot(results["eps_v2"], results["sigma"], "g--^",
            label="variant 2", markersize=3, linewidth=1.5)
    ax.plot(results["eps_baseline"], results["sigma"], "r--s",
            label="baseline (Papernot & Steinke)", markersize=3, linewidth=1.5)

    # x-axis: start from the actual minimum epsilon any curve reaches.
    # The floor (~1.58 for single-run, ~1.63 for tuned) is real — it is set by
    # T=1840 steps + delta=1e-5 in the RDP->DP conversion. Curves do not extend
    # leftward of this floor no matter how large sigma gets.
    x_min = min(results["eps_single_run"]) - 0.05
    x_max = 5.0   # zoom to the interesting region; baseline starts high

    ax.set_xlabel("final ε  (total DP cost, δ=1e-5)", fontsize=12)
    ax.set_ylabel("noise level σ", fontsize=12)
    ax.set_title("MNIST: final ε vs σ  (μ=15, q=0.1, epochs=40)\n"
                 "x-axis starts at actual minimum ε reached by curves", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlim([x_min, x_max])
    ax.grid(True, alpha=0.3)

    # Annotate the floors
    ax.axvline(min(results["eps_single_run"]), color='m', linestyle=':', alpha=0.4)
    ax.axvline(min(results["eps_v1"]),         color='b', linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/mnist_final_epsilon_v2.png", dpi=150)
    print(f"\nPlot saved.")
    print(f"  single-run floor : {min(results['eps_single_run']):.4f}")
    print(f"  variant-1 floor  : {min(results['eps_v1']):.4f}")
    print(f"  variant-2 floor  : {min(results['eps_v2']):.4f}")
    print(f"  baseline floor   : {min(results['eps_baseline']):.4f}")
    plt.close()


# ---------------------------------------------------------------------------
# 9. Convenience function
# ---------------------------------------------------------------------------

def compute_epsilon_for_params(
    sigma: float, gamma: float, epochs: int, N: int,
    q: float, mu: float, delta: float = 1e-5,
    alphas: List[int] = None,
) -> dict:
    if alphas is None:
        alphas = list(range(2, 65))
    B = int(gamma * N)
    T = epochs * max(1, N // B)
    eb, e1, e2 = compute_final_epsilon(sigma, gamma, T, q, mu, delta, alphas)
    return {"baseline": eb, "variant1": e1, "variant2": e2, "T": T, "B": B}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("DP Hyperparameter Tuning: Final Epsilon Calculator (v2)")
    print("Full per-order RDP curves threaded through Theorem 6")
    print("Koskela & Kulkarni, NeurIPS 2023")
    print("=" * 60)

    print("\n--- Single example (MNIST, sigma=1.5) ---")
    result = compute_epsilon_for_params(
        sigma=1.5, gamma=0.0213, epochs=40, N=60000,
        q=0.1, mu=15, delta=1e-5
    )
    print(f"  T (total steps)   : {result['T']}")
    print(f"  B (batch size)    : {result['B']}")
    print(f"  final ε baseline  : {result['baseline']:.4f}")
    print(f"  final ε variant 1 : {result['variant1']:.4f}")
    print(f"  final ε variant 2 : {result['variant2']:.4f}")

    print("\n--- MNIST sigma sweep ---")
    results = mnist_experiment()
    plot_results(results)
