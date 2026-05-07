import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        feat_dim=120,       # 40 MFCCs × 3 (static + Δ + ΔΔ)
        hidden_dim=256,     # wider hidden state
        num_layers=3,       # deeper network
        dropout=0.3,
    ):
        super().__init__()

        # Optional: lightweight linear projection before LSTM
        # helps when feat_dim is large
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        proj_dim = hidden_dim * 2  # bidirectional

        self.classifier = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )

    def forward(self, input_features):
        # input_features: (B, T, feat_dim)
        x = self.input_proj(input_features)   # (B, T, hidden_dim)
        enc, _ = self.encoder(x)               # (B, T, 2*hidden_dim)
        logits = self.classifier(enc)          # (B, T, num_classes)
        return logits