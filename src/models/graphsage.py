"""
graphsage.py
============
GraphSAGE model for node-level binary classification on UNSW-NB15
flow-similarity graphs.

Architecture
------------
  Input (42 features)
       ↓
  GraphSAGE layer  [in_channels → hidden_dim]
       ↓  ReLU + Dropout
  [Optional additional GraphSAGE layers  hidden_dim → hidden_dim]
       ↓  ReLU + Dropout
  Linear head  [hidden_dim → num_classes]
       ↓
  Binary prediction  (Normal=0 / Attack=1)

Notes
-----
- SAGEConv uses mean aggregation by default.
- Edge weights (cosine similarity) are NOT passed through SAGEConv here
  because SAGEConv in PyG does not accept edge weights natively in its
  mean-aggregation variant.  They are available in graph.edge_attr and
  can be incorporated in future experiments using weighted aggregation.
- The model operates at the NODE level — one prediction per flow record.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

# GNN_INPUT_DIM imported by callers from src.gnn_config


class GraphSAGE(nn.Module):
    """
    Configurable GraphSAGE network for node-level binary classification.

    Parameters
    ----------
    in_channels  : int   — number of input node features (42)
    hidden_dim   : int   — hidden dimension for all SAGE layers
    num_classes  : int   — output classes (2 for binary)
    num_layers   : int   — total number of SAGEConv layers (≥ 2)
    dropout      : float — dropout probability applied after each ReLU
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim:  int   = 64,
        num_classes: int   = 2,
        num_layers:  int   = 2,
        dropout:     float = 0.3,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.dropout = dropout

        # Build the list of SAGEConv layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim  = in_channels if i == 0 else hidden_dim
            out_dim = hidden_dim
            self.convs.append(SAGEConv(in_dim, out_dim))

        # Linear classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : node feature matrix  (N, in_channels)
        edge_index : graph connectivity   (2, E)

        Returns
        -------
        logits : (N, num_classes)
        """
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x)

    def predict_proba(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Return softmax probabilities (N, num_classes)."""
        return F.softmax(self.forward(x, edge_index), dim=-1)

    def predict(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Return class predictions (N,) — argmax of logits."""
        return self.forward(x, edge_index).argmax(dim=-1)

    # ── Convenience ───────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        layers_str = " → ".join(
            [f"SAGEConv({c.in_channels},{c.out_channels})" for c in self.convs]
        )
        return (
            f"GraphSAGE(\n"
            f"  layers : {layers_str}\n"
            f"  head   : Linear({self.classifier.in_features},{self.classifier.out_features})\n"
            f"  dropout: {self.dropout}\n"
            f"  params : {self.count_parameters():,}\n"
            f")"
        )
