"""
L1: State Encoder — 토큰을 상태 공간에 주입

Static lookup이 아니라 "상태 주입"으로 작동한다.
x_{t+1} ← x_t + E(token_t)

토큰이 들어올 때마다 현재 상태에 임펄스를 더한다.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    """토큰 → 상태 임펄스 인코더."""

    def __init__(self, vocab_size: int, d_state: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_state, padding_idx=0)
        self.d_state = d_state
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [batch, seq_len] int tensor
        Returns:
            impulses: [batch, seq_len, d_state] — 상태에 더할 임펄스
        """
        return self.embedding(token_ids)

    def encode_single(self, token_id: int) -> torch.Tensor:
        """단일 토큰 → 임펄스 벡터."""
        t = torch.tensor([token_id], dtype=torch.long)
        return self.embedding(t).squeeze(0)
