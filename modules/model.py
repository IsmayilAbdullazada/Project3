import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceClassifier(nn.Module):
    def __init__(
        self,
        num_classes,
        feat_dim=48,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        proj_dim = hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )

    def forward(self, input_features):
        enc, _ = self.encoder(input_features)
        logits = self.classifier(enc)
        return logits