"""A frozen, deterministic, offline text encoder: whitespace tokens -> a stable hash -> a fixed
random embedding table -> a fixed random per-token MLP.

This stands in for a real pretrained frozen encoder (T5, Gemma, ...) so the data pipeline,
precompute script and training loop can run without network access to a model hub. It has no
notion of language (no two tokens are related unless byte-identical) — swap in a real encoder by
registering it under a new name and pointing ``text_encoder.name`` at it (see ADR 0006); nothing
else in the data pipeline needs to change.

Deterministic across processes: token -> id uses sha256 (not Python's salted ``hash()``), and the
embedding table is seeded, so re-running precompute.py reproduces the same vectors.
"""

from __future__ import annotations

import hashlib

import torch
import torch.nn as nn

from imggen.registry import TEXT_ENCODER
from imggen.text.base import TextEncoder


def _token_id(token: str, vocab_size: int) -> int:
    digest = hashlib.sha256(token.encode()).digest()
    return int.from_bytes(digest[:4], "little") % vocab_size


@TEXT_ENCODER.register("hash")
class HashEmbeddingTextEncoder(TextEncoder):
    def __init__(
        self,
        text_dim: int = 2048,
        vocab_size: int = 32768,
        max_length: int = 32,
        seed: int = 0,
        frozen: bool = True,
    ) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.embedding = nn.Embedding(vocab_size, text_dim)
            self.mix = nn.Sequential(
                nn.Linear(text_dim, text_dim), nn.GELU(), nn.Linear(text_dim, text_dim)
            )
        if frozen:
            self.requires_grad_(False)

    def _tokenize(self, text: str) -> list[int]:
        tokens = text.strip().lower().split() or ["<empty>"]
        return [_token_id(tok, self.vocab_size) for tok in tokens[: self.max_length]]

    @torch.no_grad()
    def encode(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [self._tokenize(t) for t in texts]
        length = max(len(seq) for seq in ids)
        device = self.embedding.weight.device
        batch = torch.zeros(len(texts), length, dtype=torch.long, device=device)
        mask = torch.zeros(len(texts), length, dtype=torch.bool, device=device)
        for i, seq in enumerate(ids):
            batch[i, : len(seq)] = torch.tensor(seq, device=device)
            mask[i, : len(seq)] = True
        emb = self.mix(self.embedding(batch))
        return emb * mask[..., None], mask
