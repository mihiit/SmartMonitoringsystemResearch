"""
models/transformer_model.py
----------------------------
Transformer-based temporal risk model with optional identity embedding.

Config (Table 2 in paper):
  - 2 encoder layers
  - 4-head attention
  - d_model = 64
  - MC-Dropout p=0.20
  - Identity embedding d=16 (personalised variant)

The hybrid LSTM-Transformer (Eq. 1) combines outputs of both branches.
This module implements both standalone Transformer and the hybrid.
"""

import torch
import torch.nn as nn
import numpy as np
import math


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerRiskModel(nn.Module):
    """
    Transformer encoder for temporal disease risk.

    Parameters
    ----------
    input_dim    : number of input features (F)
    d_model      : model dimension (default 64)
    nhead        : attention heads (default 4)
    num_layers   : encoder layers (default 2)
    dropout      : MC-Dropout probability (default 0.20)
    n_patients   : if >0, enable identity embedding
    embed_dim    : identity embedding dimension (default 16)
    """

    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.20,
                 n_patients: int = 0, embed_dim: int = 16):
        super().__init__()
        self.personalised = n_patients > 0
        self.dropout_p    = dropout

        if self.personalised:
            self.patient_embed = nn.Embedding(n_patients, embed_dim)
            proj_input = input_dim + embed_dim
        else:
            proj_input = input_dim

        self.input_proj = nn.Linear(proj_input, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout  = nn.Dropout(p=dropout)
        self.fc       = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor,
                patient_ids: torch.Tensor = None) -> torch.Tensor:
        if self.personalised and patient_ids is not None:
            emb = self.patient_embed(patient_ids).unsqueeze(1).expand(-1, x.size(1), -1)
            x   = torch.cat([x, emb], dim=-1)
        x   = self.input_proj(x)
        x   = self.pos_enc(x)
        x   = self.encoder(x)          # (B, T, d_model)
        x   = self.dropout(x[:, -1])   # use [CLS]-equivalent: last token
        return torch.sigmoid(self.fc(x).squeeze(-1))

    def predict_mc(self, x: torch.Tensor, n_passes: int = 50,
                   patient_ids: torch.Tensor = None) -> tuple:
        self.train()
        preds = []
        with torch.no_grad():
            for _ in range(n_passes):
                preds.append(self.forward(x, patient_ids).unsqueeze(0))
        preds = torch.cat(preds, dim=0)
        return preds.mean(dim=0), preds.std(dim=0)


# ---------------------------------------------------------------------------
# Hybrid LSTM–Transformer (Eq. 1)
# ---------------------------------------------------------------------------

class HybridLSTMTransformer(nn.Module):
    """
    Hybrid model: R_t = f_LSTM(X_{t-k:t}) + f_Trans(X_{1:t})  [Eq. 1]

    LSTM captures short-range metabolic transitions.
    Transformer captures longer-range progression context.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 d_model: int = 64, nhead: int = 4,
                 num_lstm_layers: int = 2, num_trans_layers: int = 2,
                 dropout: float = 0.20,
                 n_patients: int = 0, embed_dim: int = 16):
        super().__init__()
        self.personalised = n_patients > 0

        if self.personalised:
            self.patient_embed = nn.Embedding(n_patients, embed_dim)
            effective_input = input_dim + embed_dim
        else:
            effective_input = input_dim

        # LSTM branch (local memory)
        self.lstm = nn.LSTM(
            input_size  = effective_input,
            hidden_size = hidden_dim,
            num_layers  = num_lstm_layers,
            batch_first = True,
            dropout     = dropout if num_lstm_layers > 1 else 0.0,
        )

        # Transformer branch (global context)
        self.input_proj = nn.Linear(effective_input, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
        enc_layer       = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_trans_layers)

        self.dropout = nn.Dropout(p=dropout)
        # Fuse both branches
        self.fc = nn.Linear(hidden_dim + d_model, 1)

    def _embed(self, x, patient_ids):
        if self.personalised and patient_ids is not None:
            emb = self.patient_embed(patient_ids).unsqueeze(1).expand(-1, x.size(1), -1)
            x   = torch.cat([x, emb], dim=-1)
        return x

    def forward(self, x: torch.Tensor,
                patient_ids: torch.Tensor = None) -> torch.Tensor:
        x_aug = self._embed(x, patient_ids)

        # LSTM branch
        lstm_out, _ = self.lstm(x_aug)
        lstm_feat   = self.dropout(lstm_out[:, -1])    # (B, hidden)

        # Transformer branch
        t_in    = self.pos_enc(self.input_proj(x_aug))
        t_out   = self.transformer(t_in)
        t_feat  = self.dropout(t_out[:, -1])           # (B, d_model)

        # Fuse and classify
        combined = torch.cat([lstm_feat, t_feat], dim=-1)
        return torch.sigmoid(self.fc(combined).squeeze(-1))

    def predict_mc(self, x: torch.Tensor, n_passes: int = 50,
                   patient_ids: torch.Tensor = None) -> tuple:
        self.train()
        preds = []
        with torch.no_grad():
            for _ in range(n_passes):
                preds.append(self.forward(x, patient_ids).unsqueeze(0))
        preds = torch.cat(preds, dim=0)
        return preds.mean(dim=0), preds.std(dim=0)


# ---------------------------------------------------------------------------
# Training utility
# ---------------------------------------------------------------------------

def train_transformer(model: nn.Module,
                      X_train: np.ndarray, y_train: np.ndarray,
                      X_val:   np.ndarray, y_val:   np.ndarray,
                      epochs: int = 50, lr: float = 5e-4,
                      batch_size: int = 64, device: str = "cpu") -> list:
    """Train Transformer/Hybrid with BCE + Adam."""
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
            bi    = idx[start:start + batch_size]
            pred  = model(X_tr[bi])
            loss  = crit(pred, y_tr[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vl = crit(model(X_v), y_v).item()
        val_losses.append(vl)

    return val_losses
