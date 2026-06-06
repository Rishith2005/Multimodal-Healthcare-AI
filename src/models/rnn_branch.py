"""
RNN Branch — ICU Time-Series Mortality / Risk Prediction
----------------------------------------------------------
LSTM-based temporal feature extractor for PhysioNet ICU vitals.
Processes variable-length sequences and extracts embeddings
from the last hidden state for fusion.
"""

import torch
import torch.nn as nn


class RNNBranch(nn.Module):
    """
    LSTM branch for ICU time-series vital signs.

    Architecture:
        LSTM(input_size=num_vitals, hidden_size=64, num_layers=2, dropout=0.3)
        → Last hidden state
        → Linear(64, embedding_dim) → ReLU → Dropout(0.2)
        → Linear(embedding_dim, num_classes)
    """

    def __init__(self, input_size: int = 12, hidden_size: int = 64,
                 num_layers: int = 2, num_classes: int = 2,
                 embedding_dim: int = 32, dropout: float = 0.3):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Embedding projection
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )

        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)

        self.embedding_dim = embedding_dim

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embedding from the last hidden state of the LSTM.

        Args:
            x: Tensor of shape (batch, seq_len, input_size)

        Returns:
            Embedding of shape (batch, embedding_dim)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state from the final layer
        last_hidden = h_n[-1]  # (batch, hidden_size)

        embedding = self.embedding_head(last_hidden)
        return embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: time-series → class logits."""
        embedding = self.get_embedding(x)
        logits = self.classifier(embedding)
        return logits


if __name__ == "__main__":
    model = RNNBranch(input_size=12, hidden_size=64, num_classes=2)
    dummy = torch.randn(8, 48, 12)  # (batch, seq_len, num_vitals)
    logits = model(dummy)
    emb = model.get_embedding(dummy)
    print(f"Logits shape: {logits.shape}")       # (8, 2)
    print(f"Embedding shape: {emb.shape}")       # (8, 32)
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
