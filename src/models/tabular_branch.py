"""
Tabular Branch — Diabetes / Pregnancy Risk Classification
-----------------------------------------------------------
Multi-layer perceptron with batch normalization and dropout.
Supports both classification and embedding extraction for fusion.
"""

import torch
import torch.nn as nn


class TabularBranch(nn.Module):
    """
    MLP branch for tabular health data.

    Architecture:
        Input(num_features)
        → Linear(64) → BatchNorm → ReLU → Dropout(0.3)
        → Linear(32) → BatchNorm → ReLU → Dropout(0.2)
        → Linear(num_classes)
    """

    def __init__(self, num_features: int = 8, num_classes: int = 2,
                 hidden_dims: list = None, embedding_dim: int = 32,
                 dropout: list = None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]
        if dropout is None:
            dropout = [0.3, 0.2]

        # Build embedding layers
        layers = []
        in_dim = num_features

        for i, (h_dim, drop) in enumerate(zip(hidden_dims, dropout)):
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=drop),
            ])
            in_dim = h_dim

        self.embedding_layers = nn.Sequential(*layers)

        # Classification head
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

        self.embedding_dim = hidden_dims[-1]

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embedding without classification head."""
        return self.embedding_layers(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: tabular features → class logits."""
        embedding = self.get_embedding(x)
        logits = self.classifier(embedding)
        return logits


if __name__ == "__main__":
    model = TabularBranch(num_features=8, num_classes=2)
    dummy = torch.randn(16, 8)
    logits = model(dummy)
    emb = model.get_embedding(dummy)
    print(f"Logits shape: {logits.shape}")       # (16, 2)
    print(f"Embedding shape: {emb.shape}")       # (16, 32)
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
