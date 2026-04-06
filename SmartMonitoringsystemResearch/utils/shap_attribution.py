"""
utils/shap_attribution.py
--------------------------
SHAP feature attribution and intervention-induced attribution shift.

Implements:
  - Mean |SHAP| per feature (Table 5 in paper)
  - ΔΦ_k = (1/T) Σ_t |φ'_k_t - φ_k_t|  (attribution shift under intervention)
  - Uses shap.DeepExplainer or shap.KernelExplainer as fallback
"""

import numpy as np
import torch
from typing import Dict, List

FEATURE_NAMES_PIMA = ["Glucose", "Blood Pressure", "BMI", "Activity", "CV Risk", "Age Factor"]
FEATURE_NAMES_NHANES = ["Glucose", "HbA1c", "BMI", "Age", "SBP", "HDL", "Activity", "Smoking"]
FEATURE_NAMES_D130 = [f"Feature_{i}" for i in range(14)]
FEATURE_NAMES_UCI = ["Glucose", "Insulin", "Exercise", "Meal", "BP", "HR"]


def compute_shap_values(model, X_background: np.ndarray,
                        X_test: np.ndarray,
                        feature_names: List[str] = None,
                        n_background: int = 100,
                        device: str = "cpu") -> np.ndarray:
    """
    Compute SHAP values using DeepExplainer (PyTorch) or KernelExplainer.

    Parameters
    ----------
    model        : trained PyTorch model
    X_background : (N_bg, T, F) background samples
    X_test       : (N_test, T, F) test samples
    feature_names: list of feature name strings
    n_background : number of background samples to use
    device       : torch device

    Returns
    -------
    shap_values : (N_test, T, F) SHAP values
    """
    try:
        import shap

        # Use a subset of background samples
        bg = X_background[:n_background]
        bg_t = torch.tensor(bg, dtype=torch.float32, device=device)

        def model_wrapper(x_np):
            model.eval()
            with torch.no_grad():
                x_t = torch.tensor(x_np.astype(np.float32), device=device)
                return model(x_t).cpu().numpy()

        # Try DeepExplainer
        try:
            explainer = shap.DeepExplainer(model, bg_t)
            x_t = torch.tensor(X_test, dtype=torch.float32, device=device)
            shap_vals = explainer.shap_values(x_t)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            return np.array(shap_vals)
        except Exception:
            # Fallback: KernelExplainer on flattened input
            X_bg_flat   = bg.reshape(len(bg), -1)
            X_test_flat = X_test.reshape(len(X_test), -1)

            def flat_wrapper(x_flat):
                N = x_flat.shape[0]
                T, F = X_test.shape[1], X_test.shape[2]
                x_3d = x_flat.reshape(N, T, F).astype(np.float32)
                return model_wrapper(x_3d)

            explainer   = shap.KernelExplainer(flat_wrapper, X_bg_flat[:50])
            shap_flat   = explainer.shap_values(X_test_flat[:20], nsamples=100)
            N, T, F     = X_test.shape
            return np.array(shap_flat).reshape(-1, T, F)

    except ImportError:
        print("[SHAP] shap not installed – returning surrogate attributions.")
        return _surrogate_shap(model, X_test, device)


def _surrogate_shap(model, X_test: np.ndarray, device: str = "cpu") -> np.ndarray:
    """
    Gradient-based surrogate attribution when SHAP is unavailable.
    Uses input × gradient (Integrated Gradients proxy).
    """
    model.eval()
    x_t  = torch.tensor(X_test, dtype=torch.float32, device=device, requires_grad=True)
    out  = model(x_t)
    out.sum().backward()
    grads = x_t.grad.detach().cpu().numpy()
    return X_test * grads


def mean_abs_shap(shap_values: np.ndarray,
                  feature_names: List[str] = None) -> Dict[str, float]:
    """
    Compute mean |SHAP| per feature, averaged over patients and time.

    Parameters
    ----------
    shap_values  : (N, T, F)
    feature_names: list of F feature names

    Returns
    -------
    Dict {feature_name: mean_abs_shap}
    """
    # Mean over N and T → (F,)
    mean_abs = np.abs(shap_values).mean(axis=(0, 1))
    F = mean_abs.shape[0]
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(F)]
    return dict(zip(feature_names[:F], mean_abs[:len(feature_names)].tolist()))


def attribution_shift(shap_baseline: np.ndarray,
                      shap_interv: np.ndarray,
                      feature_names: List[str] = None) -> Dict[str, float]:
    """
    ΔΦ_k = (1/T) Σ_t |φ'_k_t - φ_k_t|

    Per-feature attribution shift under intervention.

    Parameters
    ----------
    shap_baseline, shap_interv : (N, T, F)

    Returns
    -------
    Dict {feature_name: delta_phi}
    """
    delta = np.abs(shap_interv - shap_baseline).mean(axis=(0, 1))  # (F,)
    F = delta.shape[0]
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(F)]
    return dict(zip(feature_names[:F], delta[:len(feature_names)].tolist()))
