"""
models/lstm_model.py
--------------------
2-layer LSTM with MC-Dropout and optional identity embedding.

Config (Table 2 in paper):
  - 2 layers, hidden=64
  - MC-Dropout p=0.20
  - User embedding d=16 (personalised variant)
"""

import torch
import torch.nn as nn
import numpy as np


class LSTMRiskModel(nn.Module):
    """
    2-layer LSTM temporal risk model.

    Parameters
    ----------
    input_dim    : number of input features (F)
    hidden_dim   : LSTM hidden size (default 64)
    num_layers   : number of LSTM layers (default 2)
    dropout      : MC-Dropout probability (default 0.20)
    n_patients   : if >0, enable identity embedding (personalised variant)
    embed_dim    : identity embedding dimension (default 16)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.20,
                 n_patients: int = 0, embed_dim: int = 16):
        super().__init__()
        self.personalised = n_patients > 0
        self.dropout_p    = dropout

        if self.personalised:
            self.patient_embed = nn.Embedding(n_patients, embed_dim)
            lstm_input = input_dim + embed_dim
        else:
            lstm_input = input_dim

        self.lstm = nn.LSTM(
            input_size  = lstm_input,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor,
                patient_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x           : (batch, T, F)
        patient_ids : (batch,) LongTensor — required if personalised

        Returns
        -------
        risk : (batch,) sigmoid-activated risk score
        """
        if self.personalised and patient_ids is not None:
            emb = self.patient_embed(patient_ids)            # (B, embed_dim)
            emb = emb.unsqueeze(1).expand(-1, x.size(1), -1)
            x   = torch.cat([x, emb], dim=-1)

        out, _ = self.lstm(x)          # (B, T, hidden)
        out     = self.dropout(out[:, -1, :])   # use last time step
        logit   = self.fc(out).squeeze(-1)
        return torch.sigmoid(logit)

    def predict_mc(self, x: torch.Tensor, n_passes: int = 50,
                   patient_ids: torch.Tensor = None) -> tuple:
        """
        MC-Dropout inference: N stochastic forward passes with dropout ON.

        Returns
        -------
        mean_risk : (batch,)
        std_risk  : (batch,)
        """
        self.train()  # keeps dropout active
        preds = []
        with torch.no_grad():
            for _ in range(n_passes):
                preds.append(self.forward(x, patient_ids).unsqueeze(0))
        preds = torch.cat(preds, dim=0)   # (n_passes, batch)
        return preds.mean(dim=0), preds.std(dim=0)


# ---------------------------------------------------------------------------
# Training utility
# ---------------------------------------------------------------------------

def train_lstm(model: LSTMRiskModel,
               X_train: np.ndarray, y_train: np.ndarray,
               X_val:   np.ndarray, y_val:   np.ndarray,
               epochs: int = 50, lr: float = 1e-3,
               batch_size: int = 64, device: str = "cpu") -> list:
    """
    Train LSTM with binary cross-entropy + Adam.

    Returns list of validation losses per epoch.
    """
    model = model.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    crit  = nn.BCELoss()

    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_v  = torch.tensor(X_val,   dtype=torch.float32, device=device)
    y_v  = torch.tensor(y_val,   dtype=torch.float32, device=device)

    val_losses = []
    n = len(X_tr)
    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(n)
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            pred = model(X_tr[batch_idx])
            loss = crit(pred, y_tr[batch_idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = crit(val_pred, y_v).item()
        val_losses.append(val_loss)

    return val_losses
