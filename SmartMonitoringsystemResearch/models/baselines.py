"""
models/baselines.py
-------------------
Five baseline models evaluated on identical splits.

1. Logistic Regression   (L2, C=1.0)
2. XGBoost               (n=100, max_depth=5, lr=0.1)
3. Standard LSTM         (single-branch, no identity embedding)
4. RETAIN-style GRU      (two-level attention, re-implemented)
5. Med-BERT-style        (masked pre-training on Diabetes-130 tokens, re-implemented)

NOTE: Results are indicative; RETAIN and Med-BERT are re-implementations.
The baselines cannot produce BR values (they generate no trajectory).
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[Baselines] xgboost not installed. XGBoost baseline unavailable.")


# ---------------------------------------------------------------------------
# 1. Logistic Regression
# ---------------------------------------------------------------------------

def make_logistic_regression():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, penalty="l2", max_iter=1000,
                                   solver="lbfgs", random_state=42)),
    ])


# ---------------------------------------------------------------------------
# 2. XGBoost
# ---------------------------------------------------------------------------

def make_xgboost():
    if not HAS_XGB:
        raise RuntimeError("xgboost is not installed.")
    return XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, verbosity=0,
    )


# ---------------------------------------------------------------------------
# 3. Standard LSTM (single-branch, no embedding)
# ---------------------------------------------------------------------------
# Imported directly from models/lstm_model.py with n_patients=0


# ---------------------------------------------------------------------------
# 4. RETAIN-style GRU (re-implemented)
# ---------------------------------------------------------------------------

class RetainStyleGRU(nn.Module):
    """
    Re-implementation of RETAIN (Choi et al. 2016).
    Two-level reverse-time attention over GRU hidden states.
    Results are indicative (not original model weights).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.20):
        super().__init__()
        self.gru_alpha = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.gru_beta  = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.attn_alpha = nn.Linear(hidden_dim, 1)
        self.attn_beta  = nn.Linear(hidden_dim, input_dim)
        self.dropout    = nn.Dropout(p=dropout)
        self.fc         = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reverse time for RETAIN
        x_rev = torch.flip(x, dims=[1])
        h_a, _ = self.gru_alpha(x_rev)
        h_b, _ = self.gru_beta(x_rev)
        # Visit-level attention
        alpha = torch.softmax(self.attn_alpha(h_a), dim=1)          # (B, T, 1)
        beta  = torch.tanh(self.attn_beta(h_b))                     # (B, T, F)
        # Context vector
        context = (alpha * beta * x_rev).sum(dim=1)                 # (B, F)
        context = self.dropout(context)
        return torch.sigmoid(self.fc(context).squeeze(-1))

    def predict_proba_np(self, X: np.ndarray, device: str = "cpu") -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32, device=device)
            return self.forward(x).cpu().numpy()


# ---------------------------------------------------------------------------
# 5. Med-BERT-style (re-implemented)
# ---------------------------------------------------------------------------

class MedBertStyle(nn.Module):
    """
    Simplified Med-BERT re-implementation.
    Uses a Transformer encoder pre-trained with masked token prediction,
    then fine-tuned for binary classification.

    This is a re-implementation; results are indicative.
    Only used on Diabetes-130 (encounter token schema).
    """

    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.20):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True,
        )
        self.encoder     = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        # Masked pre-training head (used during pre-training only)
        self.mlm_head    = nn.Linear(d_model, input_dim)
        # Classification head
        self.cls_dropout = nn.Dropout(p=dropout)
        self.cls_head    = nn.Linear(d_model, 1)
        self._pretrained = False

    def pretrain_step(self, x: torch.Tensor, mask_prob: float = 0.15):
        """One masked pre-training step. Returns MLM loss."""
        mask    = torch.rand(x.shape[:2]) < mask_prob
        x_masked = x.clone()
        x_masked[mask] = 0.0
        proj  = self.input_proj(x_masked)
        enc   = self.encoder(proj)
        recon = self.mlm_head(enc)
        loss  = ((recon - x) ** 2)[mask.unsqueeze(-1).expand_as(recon)].mean()
        return loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.input_proj(x)
        enc  = self.encoder(proj)
        out  = self.cls_dropout(enc[:, -1])
        return torch.sigmoid(self.cls_head(out).squeeze(-1))

    def predict_proba_np(self, X: np.ndarray, device: str = "cpu") -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32, device=device)
            return self.forward(x).cpu().numpy()


# ---------------------------------------------------------------------------
# Flat-feature helper (for LR and XGBoost)
# ---------------------------------------------------------------------------

def flatten_sequences(X: np.ndarray) -> np.ndarray:
    """Flatten (N, T, F) → (N, T*F) for non-sequential models."""
    return X.reshape(X.shape[0], -1)


# ---------------------------------------------------------------------------
# Training helper for PyTorch baselines
# ---------------------------------------------------------------------------

def train_torch_baseline(model: nn.Module,
                         X_train: np.ndarray, y_train: np.ndarray,
                         X_val:   np.ndarray, y_val:   np.ndarray,
                         epochs: int = 50, lr: float = 1e-3,
                         batch_size: int = 64,
                         pretrain: bool = False,
                         pretrain_epochs: int = 10,
                         device: str = "cpu") -> list:
    """Generic training loop for RETAIN / Med-BERT baselines."""
    model = model.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    crit  = nn.BCELoss()

    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_v  = torch.tensor(X_val,   dtype=torch.float32, device=device)
    y_v  = torch.tensor(y_val,   dtype=torch.float32, device=device)

    # Optional masked pre-training (Med-BERT)
    if pretrain and hasattr(model, "pretrain_step"):
        pre_opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(pretrain_epochs):
            model.train()
            loss = model.pretrain_step(X_tr)
            pre_opt.zero_grad()
            loss.backward()
            pre_opt.step()

    val_losses = []
    n = len(X_tr)
    for _ in range(epochs):
        model.train()
        idx = torch.randperm(n)
        for s in range(0, n, batch_size):
            bi   = idx[s:s + batch_size]
            pred = model(X_tr[bi])
            loss = crit(pred, y_tr[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vl = crit(model(X_v), y_v).item()
        val_losses.append(vl)

    return val_losses
