"""
Fusion Model — Multimodal Healthcare Risk Prediction
-------------------------------------------------------
Combines embeddings from CNN, Tabular, and RNN branches via
concatenation and dense layers. Outputs binary risk prediction,
4-class severity classification, and per-modality confidence scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cnn_branch import CNNBranch
from src.models.tabular_branch import TabularBranch
from src.models.rnn_branch import RNNBranch


class FusionModel(nn.Module):
    """
    Multi-modal fusion model that combines pretrained branch embeddings.

    Architecture:
        CNN embedding (128) + Tabular embedding (32) + RNN embedding (32)
        → Concatenation (192-dim)
        → Linear(192, 128) → BN → ReLU → Dropout(0.4)
        → Linear(128, 64) → BN → ReLU → Dropout(0.3)
        → Binary risk head: Linear(64, 2)
        → Severity head: Linear(64, 4)

    Per-modality confidence is computed from individual branch outputs.
    """

    def __init__(self, cnn_branch: CNNBranch, tabular_branch: TabularBranch,
                 rnn_branch: RNNBranch, freeze_branches: bool = True,
                 hidden_dims: list = None, dropout: list = None,
                 num_severity_classes: int = 4):
        super().__init__()

        self.cnn_branch = cnn_branch
        self.tabular_branch = tabular_branch
        self.rnn_branch = rnn_branch

        # Optionally freeze pretrained branch weights
        if freeze_branches:
            for branch in [self.cnn_branch, self.tabular_branch, self.rnn_branch]:
                for param in branch.parameters():
                    param.requires_grad = False

        if hidden_dims is None:
            hidden_dims = [128, 64]
        if dropout is None:
            dropout = [0.4, 0.3]

        # Total embedding dimension
        total_embed_dim = (
            cnn_branch.embedding_dim +
            tabular_branch.embedding_dim +
            rnn_branch.embedding_dim
        )

        # Fusion layers
        fusion_layers = []
        in_dim = total_embed_dim

        for h_dim, drop in zip(hidden_dims, dropout):
            fusion_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=drop),
            ])
            in_dim = h_dim

        self.fusion_head = nn.Sequential(*fusion_layers)

        # Binary risk head (Low / High)
        self.binary_head = nn.Linear(hidden_dims[-1], 2)

        # Severity head (Low / Moderate / High / Critical)
        self.severity_head = nn.Linear(hidden_dims[-1], num_severity_classes)

    def get_fused_embedding(self, xray_input, tabular_input, timeseries_input):
        """Extract the fused embedding from all three branches."""
        cnn_emb = self.cnn_branch.get_embedding(xray_input)
        tab_emb = self.tabular_branch.get_embedding(tabular_input)
        rnn_emb = self.rnn_branch.get_embedding(timeseries_input)

        fused = torch.cat([cnn_emb, tab_emb, rnn_emb], dim=1)
        fused = self.fusion_head(fused)
        return fused, cnn_emb, tab_emb, rnn_emb

    def forward(self, xray_input, tabular_input, timeseries_input):
        """
        Full forward pass through all branches and fusion.

        Returns:
            dict with keys:
                - binary_logits: (B, 2) — Low / High risk
                - severity_logits: (B, 4) — Low / Moderate / High / Critical
                - modality_confidence: dict with CNN, Tabular, RNN softmax scores
        """
        fused, cnn_emb, tab_emb, rnn_emb = self.get_fused_embedding(
            xray_input, tabular_input, timeseries_input
        )

        binary_logits = self.binary_head(fused)
        severity_logits = self.severity_head(fused)

        # Per-modality confidence from individual branch classifiers
        with torch.no_grad():
            cnn_conf = F.softmax(self.cnn_branch.classifier(cnn_emb), dim=1)
            tab_conf = F.softmax(self.tabular_branch.classifier(tab_emb), dim=1)
            rnn_conf = F.softmax(self.rnn_branch.classifier(rnn_emb), dim=1)

        return {
            'binary_logits': binary_logits,
            'severity_logits': severity_logits,
            'modality_confidence': {
                'cnn': cnn_conf,
                'tabular': tab_conf,
                'rnn': rnn_conf,
            }
        }


if __name__ == "__main__":
    # Quick smoke test
    cnn = CNNBranch(num_classes=2, embedding_dim=128)
    tab = TabularBranch(num_features=8, num_classes=2)
    rnn = RNNBranch(input_size=12, num_classes=2)

    fusion = FusionModel(cnn, tab, rnn, freeze_branches=True)

    xray = torch.randn(4, 3, 224, 224)
    tabular = torch.randn(4, 8)
    timeseries = torch.randn(4, 48, 12)

    output = fusion(xray, tabular, timeseries)
    print(f"Binary logits: {output['binary_logits'].shape}")        # (4, 2)
    print(f"Severity logits: {output['severity_logits'].shape}")    # (4, 4)
    print(f"CNN confidence: {output['modality_confidence']['cnn'].shape}")  # (4, 2)

    trainable = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    total = sum(p.numel() for p in fusion.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")
