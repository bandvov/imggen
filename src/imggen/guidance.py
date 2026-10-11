"""Classifier-free guidance (CFG): drop captions during training so the model learns both a
conditional and an unconditional prediction, then at sampling time push the prediction further
in the direction the caption moves it (Ho & Salimans, 2022).

The "null" (unconditional) embedding is the text encoder's embedding of the empty string, saved
once by ``precompute.py`` (``imggen.data.shards.save_null_embedding``) — training and sampling
both reuse that saved tensor rather than loading the text encoder again.
"""

from __future__ import annotations

import torch


def apply_caption_dropout(
    text_emb: torch.Tensor,
    mask: torch.Tensor,
    null_emb: torch.Tensor,
    null_mask: torch.Tensor,
    p: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """With probability ``p``, replaces each batch element's caption embedding with the null
    embedding. Output length is ``max(text_emb.shape[1], null_emb.shape[0])`` — both the real
    and null embeddings are re-padded into fresh tensors of that length, so this works whichever
    is longer (normally the real caption, but nothing here assumes it)."""
    if p <= 0:
        return text_emb, mask
    b, dim = text_emb.shape[0], text_emb.shape[-1]
    length = max(text_emb.shape[1], null_emb.shape[0])
    drop = torch.rand(b, generator=generator) < p

    out_emb = torch.zeros(b, length, dim, dtype=text_emb.dtype)
    out_mask = torch.zeros(b, length, dtype=torch.bool)
    real_len, null_len = text_emb.shape[1], null_emb.shape[0]
    out_emb[~drop, :real_len] = text_emb[~drop]
    out_mask[~drop, :real_len] = mask[~drop]
    out_emb[drop, :null_len] = null_emb.to(text_emb.dtype)
    out_mask[drop, :null_len] = null_mask
    return out_emb, out_mask


def combine_predictions(
    cond: torch.Tensor, uncond: torch.Tensor, guidance_scale: float
) -> torch.Tensor:
    """``uncond + scale * (cond - uncond)``. ``scale=1`` recovers plain conditional generation;
    ``scale>1`` (the common case) trades diversity for stronger adherence to the caption."""
    return uncond + guidance_scale * (cond - uncond)


def broadcast_null_embedding(
    null_emb: torch.Tensor, null_mask: torch.Tensor, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expands the single saved null embedding (shape (L, D) / (L,)) to a batch."""
    return null_emb.unsqueeze(0).expand(batch_size, -1, -1), null_mask.unsqueeze(0).expand(
        batch_size, -1
    )
