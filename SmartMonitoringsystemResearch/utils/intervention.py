"""
utils/intervention.py
---------------------
Lifestyle intervention simulation as per Eq. 2 in the paper.

    act_interv(t) = act_base(t) + Δ · (1 − exp(−(t − t_start) / τ))

Parameters calibrated to the Diabetes Prevention Program (DPP) trial:
  Δ = 0.30  (magnitude – 58% incidence reduction over 3 years)
  τ = 5.0   (time constant in steps)
  t_start = 10

The function generates *two diverging trajectory branches* from a
common starting state. Branch divergence is built into the data
construction (not post-hoc input perturbation), which avoids
underestimating long-term divergence.
"""

import numpy as np


def intervention_ramp(t: np.ndarray, t_start: float = 10.0,
                      delta: float = 0.30, tau: float = 5.0) -> np.ndarray:
    """
    Compute the additive activity increment at each time step.

    Parameters
    ----------
    t        : array of time indices
    t_start  : intervention start time
    delta    : maximum increment (DPP-calibrated)
    tau      : time constant (steps)

    Returns
    -------
    increment : array, same shape as t; zero before t_start.
    """
    increment = np.where(
        t >= t_start,
        delta * (1.0 - np.exp(-(t - t_start) / tau)),
        0.0,
    )
    return increment


def make_intervention_branch(X: np.ndarray,
                             activity_feature_idx: int = 3,
                             t_start: int = 10,
                             delta: float = 0.30,
                             tau: float = 5.0) -> np.ndarray:
    """
    Create the intervention branch of a sequence batch.

    For each patient and each time step t >= t_start, the activity
    feature is incremented by the DPP ramp (clipped to [0,1]).

    Parameters
    ----------
    X                    : np.ndarray, shape (N, T, F)
    activity_feature_idx : column index of the activity feature
    t_start, delta, tau  : intervention parameters (see paper §III-A)

    Returns
    -------
    X_interv : np.ndarray, shape (N, T, F) — intervention branch
    """
    X_interv = X.copy()
    N, T, F = X.shape
    t_arr = np.arange(T, dtype=float)
    ramp  = intervention_ramp(t_arr, t_start=t_start, delta=delta, tau=tau)

    for t in range(T):
        X_interv[:, t, activity_feature_idx] = np.clip(
            X[:, t, activity_feature_idx] + ramp[t], 0.0, 1.0
        )
    return X_interv


def compute_glucose_divergence(X_base: np.ndarray,
                               X_interv: np.ndarray,
                               glucose_feature_idx: int = 0) -> np.ndarray:
    """
    Compute per-time-step mean glucose divergence (for Fig. 4).

    Returns array of shape (T,).
    """
    return (X_base[:, :, glucose_feature_idx] -
            X_interv[:, :, glucose_feature_idx]).mean(axis=0)
