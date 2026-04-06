"""
evaluation/calibration.py
--------------------------
Calibration and AUC evaluation utilities.

Implements:
  - AUC with bootstrap 95% CI (N=1000)
  - ECE (Expected Calibration Error) across 10 equal-width bins
  - Reliability diagram data
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Tuple, List, Dict


# ---------------------------------------------------------------------------
# AUC with bootstrap CI
# ---------------------------------------------------------------------------

def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC. Returns 0.5 if only one class present."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray,
                     n_bootstrap: int = 1000,
                     alpha: float = 0.05,
                     seed: int = 42) -> Tuple[float, float, float]:
    """
    Bootstrap 95% CI for AUC (N=1000 resamples).

    Returns
    -------
    (auc, lower, upper)
    """
    rng = np.random.default_rng(seed)
    N   = len(y_true)
    auc_samples = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        auc_samples.append(roc_auc_score(yt, ys))

    auc_samples = np.array(auc_samples)
    lo = float(np.percentile(auc_samples, 100 * alpha / 2))
    hi = float(np.percentile(auc_samples, 100 * (1 - alpha / 2)))
    return compute_auc(y_true, y_score), lo, hi


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray,
                n_bins: int = 10) -> float:
    """
    Expected Calibration Error across n_bins equal-width bins.

    ECE = Σ (|B_m| / N) |acc(B_m) - conf(B_m)|

    Parameters
    ----------
    y_true : binary labels
    y_prob : predicted probabilities in [0,1]
    n_bins : number of bins (paper uses 10)

    Returns
    -------
    ECE : float
    """
    bins   = np.linspace(0, 1, n_bins + 1)
    ece    = 0.0
    N      = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask   = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / N) * abs(acc - conf)

    return float(ece)


# ---------------------------------------------------------------------------
# Reliability diagram data
# ---------------------------------------------------------------------------

def reliability_diagram_data(y_true: np.ndarray, y_prob: np.ndarray,
                              n_bins: int = 10) -> Dict:
    """
    Compute data for reliability diagram (Fig. 7 in paper).

    Returns
    -------
    Dict with bin_centers, mean_prob, fraction_positives, bin_counts
    """
    bins             = np.linspace(0, 1, n_bins + 1)
    bin_centers      = (bins[:-1] + bins[1:]) / 2
    mean_prob        = np.zeros(n_bins)
    fraction_pos     = np.zeros(n_bins)
    bin_counts       = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask   = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() > 0:
            mean_prob[i]    = y_prob[mask].mean()
            fraction_pos[i] = y_true[mask].mean()
            bin_counts[i]   = mask.sum()

    return {
        "bin_centers":      bin_centers,
        "mean_prob":        mean_prob,
        "fraction_pos":     fraction_pos,
        "bin_counts":       bin_counts,
        "perfect_calib":    bin_centers,  # diagonal reference
    }


# ---------------------------------------------------------------------------
# Full evaluation summary
# ---------------------------------------------------------------------------

def evaluation_summary(y_true: np.ndarray, y_score: np.ndarray,
                       dataset_name: str = "",
                       n_bootstrap: int = 1000) -> Dict:
    """
    Compute AUC + CI + ECE for a single dataset/model.
    """
    auc, lo, hi = bootstrap_auc_ci(y_true, y_score, n_bootstrap)
    ece         = compute_ece(y_true, y_score)
    return {
        "dataset":  dataset_name,
        "AUC":      round(auc, 3),
        "CI_lower": round(lo,  3),
        "CI_upper": round(hi,  3),
        "ECE":      round(ece, 3),
        "N":        int(len(y_true)),
        "prevalence": round(float(y_true.mean()), 3),
    }
