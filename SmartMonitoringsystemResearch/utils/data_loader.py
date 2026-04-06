"""
utils/data_loader.py
--------------------
Unified data loaders for all four datasets.

Returns (X, y) pairs:
  X : np.ndarray, shape (N, T, F)   — temporal feature sequences
  y : np.ndarray, shape (N,)        — binary outcome

Loader functions are self-contained and can be called independently.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
T_DEFAULT = 30   # time steps for synthetic cohorts
T_D130    = 8    # encounter sequences for Diabetes-130


# ---------------------------------------------------------------------------
# Pima (Tier-1)
# ---------------------------------------------------------------------------

def load_pima():
    """
    Load Pima synthetic sequences.
    If pre-generated .npy files exist, load them; otherwise generate.
    """
    X_path = os.path.join(DATA_DIR, "pima_X.npy")
    y_path = os.path.join(DATA_DIR, "pima_y.npy")
    if os.path.exists(X_path) and os.path.exists(y_path):
        return np.load(X_path), np.load(y_path)
    from data.synthetic_cohorts import load_pima_static, make_pima_sequences
    df = load_pima_static()
    X, y = make_pima_sequences(df)
    np.save(X_path, X)
    np.save(y_path, y)
    return X, y


# ---------------------------------------------------------------------------
# NHANES-calibrated (Tier-1)
# ---------------------------------------------------------------------------

def load_nhanes():
    """Load NHANES-calibrated synthetic sequences."""
    X_path = os.path.join(DATA_DIR, "nhanes_X.npy")
    y_path = os.path.join(DATA_DIR, "nhanes_y.npy")
    if os.path.exists(X_path) and os.path.exists(y_path):
        return np.load(X_path), np.load(y_path)
    from data.synthetic_cohorts import make_nhanes_sequences
    X, y = make_nhanes_sequences()
    np.save(X_path, X)
    np.save(y_path, y)
    return X, y


# ---------------------------------------------------------------------------
# UCI Diabetes CGM (Tier-2, real)
# ---------------------------------------------------------------------------

def load_uci_cgm():
    """
    Load UCI Diabetes CGM dataset (N≈70, real glucose trajectories).

    Source: M. G. Kahn, UCI ML Repository ID 34, 1994.
    Outcome: poor glycaemic control (mean glucose > 154 mg/dL ≈ HbA1c 7.0%).

    If diabetes.csv is not present, falls back to a reproducible
    synthetic stand-in calibrated to the same statistics for CI testing.
    """
    csv_path = os.path.join(DATA_DIR, "uci_cgm.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return _process_uci_cgm(df)
    print("[UCI CGM] Real data not found. Using calibrated stand-in (N=70).")
    return _uci_standin()


def _process_uci_cgm(df: pd.DataFrame):
    """Process raw UCI CGM data into (X, y)."""
    # The UCI dataset has timestamped glucose, insulin, exercise, meal records.
    # We construct 6 daily features per patient.
    feature_cols = [c for c in df.columns if c.lower() not in ("id", "date", "time", "outcome")]
    feature_cols = feature_cols[:6]  # use up to 6 features

    # Group by patient if patient-id column exists
    if "patient_id" in df.columns or "id" in df.columns:
        id_col = "patient_id" if "patient_id" in df.columns else "id"
        patients = df[id_col].unique()
        N = len(patients)
        T = 30
        F = len(feature_cols)
        X = np.zeros((N, T, F), dtype=np.float32)
        y = np.zeros(N, dtype=np.int64)
        scaler = StandardScaler()
        vals = df[feature_cols].fillna(df[feature_cols].median()).values
        vals = scaler.fit_transform(vals)
        df_norm = pd.DataFrame(vals, columns=feature_cols)
        df_norm[id_col] = df[id_col].values
        for i, pid in enumerate(patients):
            rows = df_norm[df_norm[id_col] == pid][feature_cols].values
            rows = rows[:T] if len(rows) >= T else np.pad(rows, ((0, T - len(rows)), (0, 0)))
            X[i] = rows[:T]
        # Outcome: mean glucose > threshold
        if "Glucose" in df.columns or "glucose" in df.columns:
            gcol = "Glucose" if "Glucose" in df.columns else "glucose"
            means = df.groupby(id_col)[gcol].mean().values
            y = (means > 154).astype(np.int64)
    else:
        return _uci_standin()
    return X, y


def _uci_standin(n=70, T=30, F=6, seed=42):
    """
    Reproducible synthetic stand-in calibrated to UCI CGM statistics.
    Used when real data is unavailable (CI/testing only).
    """
    rng = np.random.default_rng(seed)
    labels = rng.binomial(1, 0.50, n)   # ~50% poor control in IDDM cohort
    X = np.zeros((n, T, F), dtype=np.float32)
    for i in range(n):
        base_glucose = 1.5 + 0.4 * labels[i]
        for t in range(T):
            noise = rng.normal(0, 0.1, F)
            X[i, t, 0] = base_glucose + 0.01 * t * labels[i] + noise[0]  # glucose
            X[i, t, 1] = rng.normal(0.5, 0.1) + noise[1]                 # insulin
            X[i, t, 2] = rng.normal(0.4, 0.1) + noise[2]                 # exercise
            X[i, t, 3] = rng.normal(0.5, 0.1) + noise[3]                 # meal
            X[i, t, 4] = rng.normal(0.5, 0.1) + noise[4]                 # BP
            X[i, t, 5] = rng.normal(0.5, 0.1) + noise[5]                 # HR
    X = np.clip(X, 0, 1)
    return X.astype(np.float32), labels.astype(np.int64)


# ---------------------------------------------------------------------------
# Diabetes 130-US Hospitals (Tier-2, real, primary)
# ---------------------------------------------------------------------------

def load_diabetes130(n_patients: int = 5246):
    """
    Load and process Diabetes 130-US Hospitals dataset.

    Source: B. Strack et al., BioMed Res. Int., 2014. UCI ID 296.

    Task: predict HbA1c > 7% at final encounter.
    Features: 14 per encounter (HbA1c excluded from inputs to prevent leakage).
    T = 8 encounters; 5-fold stratified CV.

    Returns (X, y) where X.shape = (N, T, 14), y.shape = (N,).
    """
    csv_path = os.path.join(DATA_DIR, "diabetes130.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, low_memory=False)
        return _process_diabetes130(df, n_patients)
    print("[Diabetes-130] Real data not found. Using calibrated stand-in.")
    return _diabetes130_standin(n_patients)


def _process_diabetes130(df: pd.DataFrame, n_patients: int = 5246):
    """Process raw Diabetes-130 CSV into (X, y)."""
    # Replace '?' with NaN
    df = df.replace("?", np.nan)

    # Map A1C result to binary outcome: >7 → 1
    if "A1Cresult" in df.columns:
        df["outcome"] = (df["A1Cresult"].isin([">7", ">8"])).astype(int)
    else:
        df["outcome"] = 0

    # Select numeric features (excluding identifiers and target)
    exclude = ["encounter_id", "patient_nbr", "A1Cresult", "outcome",
               "readmitted", "payer_code", "medical_specialty",
               "diag_1", "diag_2", "diag_3"]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in exclude][:14]

    # Fill NaN with column median
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    # Group by patient, take up to T=8 encounters
    T = T_D130
    F = len(num_cols)

    if "patient_nbr" in df.columns:
        patients = df["patient_nbr"].unique()[:n_patients]
        N = len(patients)
        X = np.zeros((N, T, F), dtype=np.float32)
        y = np.zeros(N, dtype=np.int64)
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

        for i, pid in enumerate(patients):
            rows = df[df["patient_nbr"] == pid]
            feats = rows[num_cols].values
            label = rows["outcome"].values[-1]
            if len(feats) >= T:
                X[i] = feats[-T:]
            else:
                X[i, :len(feats)] = feats
            y[i] = label
    else:
        # Flat format fallback
        scaler = StandardScaler()
        vals = scaler.fit_transform(df[num_cols].values[:n_patients])
        N = min(n_patients, len(vals))
        X = np.zeros((N, T_D130, F), dtype=np.float32)
        for i in range(N):
            X[i] = np.tile(vals[i], (T_D130, 1))
        y = df["outcome"].values[:N].astype(np.int64)

    return X, y


def _diabetes130_standin(n=5246, T=8, F=14, seed=44):
    """
    Calibrated stand-in for Diabetes-130 (CI/testing).
    Prevalence 69.1% (HbA1c > 7%).
    """
    rng = np.random.default_rng(seed)
    labels = rng.binomial(1, 0.691, n)
    X = rng.normal(0, 1, (n, T, F)).astype(np.float32)
    # Make outcome correlated with features
    X[:, :, 0] += labels[:, None] * 0.5
    return X, labels.astype(np.int64)


# ---------------------------------------------------------------------------
# Train/val/test split helpers
# ---------------------------------------------------------------------------

def stratified_split(X, y, train=0.75, val=0.10, seed=42):
    """Stratified split: 75/10/15 (Pima) or 80/20 test."""
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train), stratify=y, random_state=seed
    )
    val_frac = val / (1 - train)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_frac), stratify=y_temp, random_state=seed
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def kfold_splits(X, y, n_splits=5, seed=42):
    """Yield (train_idx, test_idx) for stratified k-fold CV."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(X, y))
