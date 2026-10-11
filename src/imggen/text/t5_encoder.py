"""Wraps a frozen pretrained T5 encoder (``transformers.T5EncoderModel``) to the ``TextEncoder``
interface — the real pretrained text encoder ADR 0006 deferred. Needs ``transformers`` and
``sentencepiece`` installed (``pip install transformers sentencepiece``) and, for
``from_pretrained``, network access to huggingface.co — not available in this project's own
sandbox (ADR 0013); the ``from_pretrained`` path is untested here, same caveat as
``imggen.autoencoders.sd_vae``. Tested via injected tiny local model/tokenizer instead
(``_model``/``_tokenizer``).
"""

from __future__ import annotations

import torch

from imggen.registry import TEXT_ENCODER
from imggen.text.base import TextEncoder

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


@TEXT_ENCODER.register("t5")
class T5TextEncoder(TextEncoder):
    def __init__(
        self,
        model_id: str = "google/t5-v1_1-small",
        max_length: int = 64,
        dtype: str = "float32",
        frozen: bool = True,
        _model=None,
        _tokenizer=None,
    ) -> None:
        """``_model``/``_tokenizer``: inject already-constructed objects (e.g. for testing
        without network access); production use leaves both ``None`` and loads ``model_id``
        from the hub."""
        super().__init__()
        if _model is None or _tokenizer is None:
            from transformers import T5EncoderModel, T5TokenizerFast

            if dtype not in _DTYPES:
                raise ValueError(f"unknown dtype {dtype!r} (expected one of {list(_DTYPES)})")
            _tokenizer = _tokenizer or T5TokenizerFast.from_pretrained(model_id)
            _model = _model or T5EncoderModel.from_pretrained(model_id, torch_dtype=_DTYPES[dtype])
        self.tokenizer = _tokenizer
        self.model = _model
        self.text_dim = self.model.config.d_model
        self.max_length = max_length
        self.model.eval()
        if frozen:
            self.requires_grad_(False)

    @torch.no_grad()
    def encode(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.model.parameters()).device
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(device)
        attn_mask = encoded["attention_mask"].to(device)
        out = self.model(input_ids=input_ids, attention_mask=attn_mask)
        return out.last_hidden_state.float(), attn_mask.bool()
