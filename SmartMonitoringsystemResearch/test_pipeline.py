"""
test_pipeline.py
----------------
Quick end-to-end smoke test for CI / reproducibility check.
Runs in < 2 minutes on CPU with --quick flag.

Usage:
    python test_pipeline.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

def test(name, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
    except Exception as e:
        print(f"  {FAIL}  {name}: {e}")
        sys.exit(1)


# 1. Synthetic cohorts
def _test_pima():
    from data.synthetic_cohorts import load_pima_static, make_pima_sequences
    df = load_pima_static()
    X, y = make_pima_sequences(df)
    assert X.shape == (768, 30, 6), f"bad shape {X.shape}"
    assert set(np.unique(y)) == {0, 1}

def _test_nhanes():
    from data.synthetic_cohorts import make_nhanes_sequences
    X, y = make_nhanes_sequences(n=200)
    assert X.shape == (200, 30, 8)

# 2. Intervention
def _test_intervention():
    from utils.intervention import make_intervention_branch, intervention_ramp
    t = np.arange(30)
    ramp = intervention_ramp(t, t_start=10)
    assert ramp[:10].sum() == 0, "ramp should be zero before t_start"
    assert ramp[11] > 0, "ramp should grow after t_start"  # ramp[10]=0 by definition (1-exp(0)=0)
    X = np.random.rand(10, 30, 6).astype(np.float32)
    X_i = make_intervention_branch(X, activity_feature_idx=3)
    # After intervention, activity should be >= baseline (clipping may equal)
    assert (X_i[:, 10:, 3] >= X[:, 10:, 3] - 1e-6).all()

# 3. BR metric
def _test_br():
    from evaluation.br_metric import (compute_br, compute_br_velocity,
                                       wilcoxon_test, sensitivity_analysis,
                                       full_br_report)
    rng = np.random.default_rng(0)
    R_base   = rng.uniform(0.3, 0.8, (100, 30)).astype(np.float32)
    R_interv = R_base - 0.022
    br = compute_br(R_base, R_interv)
    assert abs(br - 0.022) < 1e-4, f"BR={br}"
    bv = compute_br_velocity(R_base, R_interv)
    assert bv < 0.01
    _, p = wilcoxon_test(R_base, R_interv)
    assert p < 0.05
    sens = sensitivity_analysis(br)
    assert sens["within_range"]
    report = full_br_report(R_base, R_interv, "test", n_bootstrap=50)
    assert "wilcoxon_p" in report

# 4. Calibration
def _test_calibration():
    from evaluation.calibration import compute_ece, bootstrap_auc_ci, evaluation_summary
    rng = np.random.default_rng(1)
    y_true = rng.binomial(1, 0.4, 200)
    y_prob = rng.beta(2, 3, 200)
    ece = compute_ece(y_true, y_prob)
    assert 0 <= ece <= 1
    auc, lo, hi = bootstrap_auc_ci(y_true, y_prob, n_bootstrap=100)
    assert lo < auc < hi
    summary = evaluation_summary(y_true, y_prob, n_bootstrap=100)
    assert "AUC" in summary and "ECE" in summary

# 5. Uncertainty
def _test_uncertainty():
    from evaluation.uncertainty import coefficient_of_variation, uncertainty_summary
    mean_r = np.array([0.1, 0.5, 0.9])
    std_r  = np.array([0.02, 0.05, 0.03])
    cov = coefficient_of_variation(mean_r, std_r)
    assert cov[1] == pytest_approx_manual(std_r[1] / mean_r[1] * 100, rel=1e-3)
    summary = uncertainty_summary(mean_r, std_r, "test")
    assert "cov_percent" in summary

def pytest_approx_manual(val, expected=None, rel=1e-3):
    return val   # just return for assertion use

# 6. LSTM model
def _test_lstm():
    from models.lstm_model import LSTMRiskModel
    model = LSTMRiskModel(input_dim=6, hidden_dim=32, num_layers=2)
    x = torch.randn(8, 30, 6)
    out = model(x)
    assert out.shape == (8,)
    assert (out >= 0).all() and (out <= 1).all()
    mean_r, std_r = model.predict_mc(x, n_passes=10)
    assert mean_r.shape == (8,)

# 7. Transformer model
def _test_transformer():
    from models.transformer_model import TransformerRiskModel, HybridLSTMTransformer
    model = TransformerRiskModel(input_dim=6)
    x = torch.randn(8, 30, 6)
    out = model(x)
    assert out.shape == (8,)
    hybrid = HybridLSTMTransformer(input_dim=6)
    out2 = hybrid(x)
    assert out2.shape == (8,)

# 8. Baselines
def _test_baselines():
    from models.baselines import (make_logistic_regression, RetainStyleGRU,
                                   MedBertStyle, flatten_sequences)
    X = np.random.rand(50, 10, 6).astype(np.float32)
    y = np.random.randint(0, 2, 50)
    lr = make_logistic_regression()
    lr.fit(flatten_sequences(X), y)
    proba = lr.predict_proba(flatten_sequences(X))[:, 1]
    assert proba.shape == (50,)

    retain = RetainStyleGRU(input_dim=6, hidden_dim=16)
    preds = retain.predict_proba_np(X)
    assert preds.shape == (50,)

    medbert = MedBertStyle(input_dim=6, d_model=16, nhead=2)
    preds2 = medbert.predict_proba_np(X)
    assert preds2.shape == (50,)

# 9. Data loaders
def _test_data_loaders():
    from utils.data_loader import (load_pima, load_nhanes, load_uci_cgm,
                                    load_diabetes130, stratified_split, kfold_splits)
    X, y = load_pima()
    assert X.ndim == 3
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = stratified_split(X, y)
    assert len(X_tr) + len(X_val) + len(X_te) == len(X)
    splits = kfold_splits(X, y, n_splits=3)
    assert len(splits) == 3

# 10. Figures (no display, just check they run without error)
def _test_figures():
    from figures.plot_all import (fig2_auc_ci, fig3_trajectories,
                                   fig4_glucose_divergence, fig7_shap,
                                   fig9_cross_dataset)
    fig2_auc_ci()
    fig3_trajectories()
    fig4_glucose_divergence()
    fig7_shap()
    fig9_cross_dataset()


if __name__ == "__main__":
    print("\n=== BR Paper — Reproducibility Test Suite ===\n")
    test("Pima synthetic cohort",         _test_pima)
    test("NHANES synthetic cohort",        _test_nhanes)
    test("Intervention ramp (Eq. 2)",      _test_intervention)
    test("BR / BRvelocity / Wilcoxon",     _test_br)
    test("Calibration (ECE, AUC, CI)",     _test_calibration)
    test("MC-Dropout uncertainty",         _test_uncertainty)
    test("LSTM model (MC-Dropout)",        _test_lstm)
    test("Transformer / Hybrid model",     _test_transformer)
    test("Baseline models (LR, RETAIN, MedBERT)", _test_baselines)
    test("Data loaders & splits",          _test_data_loaders)
    test("Figure generation (Fig 2–9)",    _test_figures)
    print("\n  All tests passed.\n")
