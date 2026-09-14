"""LiftGCN model for nodal von Mises stress regression."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

NODE_FEATURE_DIM = 4
JOUKOWSKI_RHO_MAX = 0.99
JOUKOWSKI_RHO_INIT = 0.90
JOUKOWSKI_RES_SCALE_INIT = 0.10

class JoukowskiBlock(nn.Module):
    """Apply a node-wise nonlinear residual correction without extra graph aggregation."""

    def __init__(
        self,
        hidden: int,
        dropout: float,
        res_scale_init: float = JOUKOWSKI_RES_SCALE_INIT,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.linear = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.res_scale = nn.Parameter(torch.tensor(float(res_scale_init)))

    def forward(self, h_bar: torch.Tensor) -> torch.Tensor:
        delta = self.linear(self.norm(h_bar))
        delta = self.dropout(F.gelu(delta))
        return h_bar + self.res_scale * delta


class StressJoukowskiGCN(nn.Module):
    """LiftGCN with shared per-channel rho and second-order Joukowski propagation.

    The backbone uses H_bar_1 = A_hat H_0 R and
    H_bar_next = 2 A_hat H_current R - H_previous.
    Each step adds a learned node-wise nonlinear residual correction.
    The readout consumes both final states and returns [num_nodes, 1].
    Exact energy conservation applies to the linear backbone, not the full
    nonlinear network."""

    def __init__(
        self,
        in_dim: int = NODE_FEATURE_DIM,
        hidden: int = 128,
        layers: int = 6,
        dropout: float = 0.1,
        rho_max: float = JOUKOWSKI_RHO_MAX,
        rho_init: float = JOUKOWSKI_RHO_INIT,
        res_scale_init: float = JOUKOWSKI_RES_SCALE_INIT,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1.")
        if not 0.0 < rho_max < 1.0:
            raise ValueError("rho_max must satisfy 0 < rho_max < 1.")
        if not abs(rho_init) < rho_max:
            raise ValueError("rho_init must satisfy abs(rho_init) < rho_max.")

        self.in_dim = in_dim
        self.hidden = hidden
        self.layers = layers
        self.rho_max = float(rho_max)

        self.input = nn.Linear(in_dim, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.input_dropout = nn.Dropout(dropout)

        ratio = float(rho_init) / self.rho_max
        ratio = min(max(ratio, -1.0 + 1e-6), 1.0 - 1e-6)
        raw_init = 0.5 * math.log((1.0 + ratio) / (1.0 - ratio))
        self.raw_rho = nn.Parameter(torch.full((hidden,), raw_init, dtype=torch.float32))

        self.blocks = nn.ModuleList(
            [JoukowskiBlock(hidden, dropout, res_scale_init) for _ in range(layers)]
        )

        self.output = nn.Linear(2 * hidden, 1)

    def rho(self) -> torch.Tensor:
        return self.rho_max * torch.tanh(self.raw_rho)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h0 = self.input(x)
        h0 = self.input_dropout(F.gelu(self.input_norm(h0)))
        rho = self.rho().view(1, -1)

        h_prev = h0
        h_bar = torch.sparse.mm(adj, h0) * rho
        h_curr = self.blocks[0](h_bar)

        for block in self.blocks[1:]:
            h_bar = 2.0 * torch.sparse.mm(adj, h_curr) * rho - h_prev
            h_next = block(h_bar)
            h_prev, h_curr = h_curr, h_next

        return self.output(torch.cat([h_curr, h_prev], dim=1))



# Public model name; retain the original class name for compatibility.
LiftGCN = StressJoukowskiGCN
