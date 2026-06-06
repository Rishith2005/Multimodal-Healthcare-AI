"""
CNN Branch — Chest X-ray Classification
-----------------------------------------
ResNet18-based feature extractor with configurable embedding dimension.
Supports both classification forward pass and embedding extraction
for the fusion stage. Includes Grad-CAM target layer access.
"""

import torch
import torch.nn as nn
from torchvision import models


class CNNBranch(nn.Module):
    """
    CNN branch using a pretrained ResNet18 backbone.

    Architecture:
        ResNet18 (pretrained) → AdaptiveAvgPool → Flatten
        → Linear(512, embedding_dim) → ReLU → Dropout
        → Linear(embedding_dim, num_classes)
    """

    def __init__(self, num_classes: int = 2, embedding_dim: int = 128,
                 dropout: float = 0.3, pretrained: bool = True):
        super().__init__()

        # Load pretrained ResNet18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Store the target layer for Grad-CAM (last conv block)
        self.grad_cam_target_layer = self.backbone.layer4[-1]

        # Remove the original FC layer
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Identity()

        # Embedding head
        self.embedding_head = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)

        self.embedding_dim = embedding_dim

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embedding without classification head."""
        features = self.backbone(x)
        embedding = self.embedding_head(features)
        return embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: input image → class logits."""
        embedding = self.get_embedding(x)
        logits = self.classifier(embedding)
        return logits


if __name__ == "__main__":
    model = CNNBranch(num_classes=2, embedding_dim=128)
    dummy = torch.randn(4, 3, 224, 224)
    logits = model(dummy)
    emb = model.get_embedding(dummy)
    print(f"Logits shape: {logits.shape}")       # (4, 2)
    print(f"Embedding shape: {emb.shape}")       # (4, 128)
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
