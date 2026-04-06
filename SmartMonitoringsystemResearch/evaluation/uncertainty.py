"""
evaluation/uncertainty.py
--------------------------
MC-Dropout uncertainty estimation.

Implements:
  - N=50 stochastic forward passes with dropout retained at inference
  - CoV = σ / R̂ (scale-normalised stability measure)
  - Population-level and per-patient uncertainty summary
"""

import numpy as np
import torch
from typing import Dict, Tuple


def mc_dropout_predict(model, X: np.ndarray,
                       n_passes: int = 50,
                       device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """
    MC-Dropout inference: N stochastic forward passes.

    Dropout is kept ACTIVE during inference (model.train() mode).

    Parameters
    ----------
    model    : any model with .forward() → (N,) risk scores
    X        : (N, T, F)
    n_passes : number of stochastic samples (paper uses 50)
    device   : torch device string

    Returns
    -------
    mean_risk : (N,) — mean predicted risk across passes
    std_risk  : (N,) — std of predicted risk across passes
    """
    if hasattr(model, "predict_mc"):
        x_t = torch.tensor(X, dtype=torch.float32, device=device)
        mean_r, std_r = model.predict_mc(x_t, n_passes=n_passes)
        return mean_r.cpu().numpy(), std_r.cpu().numpy()

    # Generic fallback
    model.train()  # keeps dropout active
    preds = []
    with torch.no_grad():
        x_t = torch.tensor(X, dtype=torch.float32, device=device)
        for _ in range(n_passes):
            out = model(x_t)
            preds.append(out.cpu().numpy())
    preds = np.stack(preds, axis=0)    # (n_passes, N)
    return preds.mean(axis=0), preds.std(axis=0)


def coefficient_of_variation(mean_risk: np.ndarray,
                              std_risk: np.ndarray) -> np.ndarray:
    """
    CoV = σ / R̂  (per-patient, in percent).

    Patients with mean_risk ≈ 0 are assigned CoV = NaN.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = np.where(mean_risk > 1e-6, std_risk / mean_risk * 100, np.nan)
    return cov


def uncertainty_summary(mean_risk: np.ndarray,
                        std_risk:  np.ndarray,
                        dataset_name: str = "") -> Dict:
    """
    Compute population-level uncertainty summary (Table 6 in paper).

    Returns
    -------
    Dict matching Table 6 format.
    """
    cov = coefficient_of_variation(mean_risk, std_risk)
    cov_valid = cov[~np.isnan(cov)]

    # Per-patient examples (lowest and highest risk)
    sorted_idx = np.argsort(mean_risk)
    low_idx    = sorted_idx[0]
    high_idx   = sorted_idx[-1]

    return {
        "dataset":                   dataset_name,
        "mean_predicted_risk":       round(float(mean_risk.mean()), 4),
        "mc_dropout_sigma_pop_mean": round(float(std_risk.mean()),  4),
        "confidence_1_minus_sigma":  round(float((1 - std_risk.mean()) * 100), 2),
        "cov_percent":               round(float(cov_valid.mean()),  2),
        "n_passes":                  50,
        "patient_examples": {
            f"patient_{high_idx}_risk":  round(float(mean_risk[high_idx]), 3),
            f"patient_{high_idx}_cov":   round(float(cov[high_idx]), 2),
            f"patient_{low_idx}_risk":   round(float(mean_risk[low_idx]), 3),
            f"patient_{low_idx}_cov":    round(float(cov[low_idx]), 2)
            if not np.isnan(cov[low_idx]) else "N/A",
        },
    }
