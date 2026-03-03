import torch
import torch.nn as nn


class LoRA(nn.Module):
    """Low-Rank Adaptation for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        device=None,
    ):
        super().__init__()
        self.device = device if device else torch.device("cuda:0")
        self.qkv = qkv.to(self.device)
        self.linear_a_q = linear_a_q.to(self.device)
        self.linear_b_q = linear_b_q.to(self.device)
        self.linear_a_v = linear_a_v.to(self.device)
        self.linear_b_v = linear_b_v.to(self.device)
        self.dim = qkv.in_features

    def forward(self, x) -> torch.Tensor:
        x = x.to(self.device)

        # Compute the original qkv
        qkv = self.qkv(x)  # Shape: (B, N, 3 * org_C)

        # Compute the new q and v components
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))

        # Add new q and v components to the original qkv tensor
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v

        return qkv
