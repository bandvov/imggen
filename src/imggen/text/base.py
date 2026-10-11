"""Text encoder interface: strings in, per-token features and a padding mask out.

    encode(texts: list[str]) -> (embeddings (B, L, text_dim), mask (B, L) bool, True = real token)

``L`` is the batch's own max token count (or truncated to ``max_length``); embeddings are padded,
never the other way around. ``text_dim`` is read by the data pipeline to check it matches
``backbone.text_dim``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class TextEncoder(nn.Module, ABC):
    text_dim: int

    @abstractmethod
    def encode(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]: ...
