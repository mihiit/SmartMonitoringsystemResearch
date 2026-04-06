"""
evaluation/br_metric.py
------------------------
Behavioural Robustness (BR) evaluation framework.

Implements:
  - BR  (Eq. 3): absolute trajectory divergence
  - BRvelocity (Eq. 4): rate-of-change divergence
  - Wilcoxon signed-rank test on {R'_t - R_t}
  - Bootstrap 95% CI on BR
  - Sensitivity analysis across reference range [0.01, 0.05]
  - Multi-horizon risk prediction (7, 30, 90 day)

Reference range for BR interpretation:
  ≈ 0.01–0.03, derived from DPP trial 58% incidence reduction [Knowler et al., 2002]
  BR ≈ 0     → intervention insensitivity (failure mode)
  BR > 0.10  → implausible instantaneous reversal (failure mode)
"""

import numpy as np
from scipy.stats import wilcoxon
from typing import Tuple, Dict


# ---------------------------------------------------------------------------
# Core BR metrics
# ---------------------------------------------------------------------------

def compute_br(R_baseline: np.ndarray, R_intervention: np.ndarray) -> float:
    """
    BR: mean absolute trajectory divergence (Eq. 3).

    Parameters
    ----------
    R_baseline     : (N, T) — predicted risk under baseline
    R_intervention : (N, T) — predicted risk under intervention

    Returns
    -------
    BR : float — scalar mean absolute divergence
    """
    return float(np.mean(np.abs(R_intervention - R_baseline)))


def compute_br_velocity(R_baseline: np.ndarray,
                        R_intervention: np.ndarray,
                        dt: float = 1.0) -> float:
    """
    BRvelocity: mean absolute rate-of-change divergence (Eq. 4).

    V_t = (R_t - R_{t-1}) / Δt

    Parameters
    ----------
    R_baseline, R_intervention : (N, T)
    dt                         : time step size (default 1.0)

    Returns
    -------
    BRvelocity : float
    """
    V_base  = np.diff(R_baseline,  axis=1) / dt  # (N, T-1)
    V_interv= np.diff(R_intervention, axis=1) / dt
    return float(np.mean(np.abs(V_interv - V_base)))


def compute_mean_delta_r(R_baseline: np.ndarray,
                         R_intervention: np.ndarray) -> float:
    """Mean signed divergence R'_t - R_t (positive = intervention raised risk)."""
    return float(np.mean(R_intervention - R_baseline))


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test
# ---------------------------------------------------------------------------

def wilcoxon_test(R_baseline: np.ndarray,
                  R_intervention: np.ndarray) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test on {R'_t - R_t} pooled over patients and time.

    No distributional assumptions are made [Rosenbaum 2002].

    Returns
    -------
    statistic : float
    p_value   : float
    """
    diff = (R_intervention - R_baseline).flatten()
    # Remove ties (zero differences)
    diff = diff[diff != 0.0]
    if len(diff) == 0:
        return 0.0, 1.0
    stat, p = wilcoxon(diff, alternative="two-sided")
    return float(stat), float(p)


# ---------------------------------------------------------------------------
# Bootstrap CI for BR
# ---------------------------------------------------------------------------

def bootstrap_br_ci(R_baseline: np.ndarray,
                    R_intervention: np.ndarray,
                    n_bootstrap: int = 1000,
                    alpha: float = 0.05,
                    seed: int = 42) -> Tuple[float, float]:
    """
    Bootstrap 95% CI for BR (N=1000 resamples).

    Returns
    -------
    (lower, upper) : float tuple
    """
    rng = np.random.default_rng(seed)
    N   = R_baseline.shape[0]
    br_samples = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        br_samples.append(compute_br(R_baseline[idx], R_intervention[idx]))
    lo = float(np.percentile(br_samples, 100 * alpha / 2))
    hi = float(np.percentile(br_samples, 100 * (1 - alpha / 2)))
    return lo, hi


# ---------------------------------------------------------------------------
# Trajectory extraction from trained model
# ---------------------------------------------------------------------------

def extract_trajectories(model, X: np.ndarray,
                         device: str = "cpu") -> np.ndarray:
    """
    Run model over all time steps of X to produce R_t sequence.

    For a model trained to predict 7-day risk from input up to t,
    we slide a growing window to build the full trajectory.

    Parameters
    ----------
    model : any model with forward() that returns (N,) risk scores
    X     : (N, T, F)
    device: torch device string

    Returns
    -------
    R : (N, T) — risk trajectory
    """
    import torch
    N, T, F = X.shape
    R = np.zeros((N, T), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for t in range(1, T + 1):
            x_t = torch.tensor(X[:, :t, :], dtype=torch.float32, device=device)
            # Pad to full T if model expects fixed length
            if t < T:
                pad = torch.zeros(N, T - t, F, dtype=torch.float32, device=device)
                x_t = torch.cat([pad, x_t], dim=1)
            out = model(x_t)
            R[:, t - 1] = out.cpu().numpy()
    return R


# ---------------------------------------------------------------------------
# Multi-horizon risk (7 / 30 / 90 day)
# ---------------------------------------------------------------------------

def multi_horizon_risk(R: np.ndarray, T: int,
                       horizons=(7, 30, 90)) -> Dict[int, float]:
    """
    Extract mean risk at specific horizons (indices scaled to T).

    Returns dict {horizon: mean_risk}.
    """
    result = {}
    for h in horizons:
        idx = min(h - 1, T - 1)
        result[h] = float(R[:, idx].mean())
    return result


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def sensitivity_analysis(BR: float, reference_range=(0.01, 0.05)) -> Dict:
    """
    Assess BR relative to the DPP-anchored reference range.

    Returns a dict with interpretation keys.
    """
    lo, hi = reference_range
    return {
        "BR": BR,
        "reference_range": reference_range,
        "within_range": lo <= BR <= hi,
        "below_range":  BR < lo,
        "above_range":  BR > hi,
        "implausible_reversal": BR > 0.10,
        "insensitive": BR < 0.005,
        "interpretation": (
            "Within DPP-anchored reference range (physiologically consistent)"
            if lo <= BR <= hi else
            "Below reference range (possible intervention insensitivity)"
            if BR < lo else
            "Above reference range — check for implausible reversal" if BR > 0.10
            else "Slightly above range — monitor"
        ),
    }


# ---------------------------------------------------------------------------
# Full BR evaluation report
# ---------------------------------------------------------------------------

def full_br_report(R_baseline: np.ndarray,
                   R_intervention: np.ndarray,
                   dataset_name: str = "",
                   n_bootstrap: int = 1000) -> Dict:
    """
    Compute and return all BR metrics for a dataset.

    Returns
    -------
    Dict with all BR metrics, Wilcoxon results, CI, and interpretation.
    """
    br          = compute_br(R_baseline, R_intervention)
    br_vel      = compute_br_velocity(R_baseline, R_intervention)
    mean_dr     = compute_mean_delta_r(R_baseline, R_intervention)
    stat, pval  = wilcoxon_test(R_baseline, R_intervention)
    ci_lo, ci_hi= bootstrap_br_ci(R_baseline, R_intervention, n_bootstrap)
    sensitivity = sensitivity_analysis(br)

    report = {
        "dataset":          dataset_name,
        "BR":               round(br, 4),
        "BRvelocity":       round(br_vel, 4),
        "mean_delta_R":     round(mean_dr, 4),
        "wilcoxon_stat":    round(stat, 2),
        "wilcoxon_p":       pval,
        "br_ci_95":         (round(ci_lo, 4), round(ci_hi, 4)),
        "sensitivity":      sensitivity,
    }
    return report
