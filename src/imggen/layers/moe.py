"""Sparse mixture-of-experts FFN (dropless, top-k routing, optional always-on shared FFN).

Routing: softmax over all experts, pick top-k, renormalise the k probabilities to sum to 1 (so
k=1 gives weight exactly 1 and a 1-expert MoE equals a dense FFN). Routing math runs in fp32
even under bf16 autocast.

Load balancing (``moe.balance``):
  aux_loss  Switch-style loss E * sum_e f_e * P_e, stored (already weighted) in ``aux_loss``.
            The training loop adds ``collect_aux_loss(model)`` to its loss.
  bias      loss-free: a per-expert selection bias (buffer ``route_bias``) nudged towards
            balance by ``step_router_bias(model)``, called once per optimizer step. The bias only
            affects which experts are chosen, never the combine weights. Keep the model in fp32
            (bf16 via autocast), since a bf16 bias buffer would swallow the small updates.
            Multi-rank runs must all-reduce ``load_accum`` first (step 8).
  none      no balancing.

Side channels after each forward: ``last_load`` ((E,) fraction of assignments per expert) and,
in training mode, ``aux_loss`` / ``load_accum``. Experts are stacked parameters so they can be
sharded (FSDP / expert parallelism) later; dispatch is a per-expert loop over sorted tokens.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from imggen.config import MoEConfig
from imggen.layers.ffn import DenseFFN
from imggen.registry import FFN


@FFN.register("moe")
class MoEFFN(nn.Module):
    def __init__(self, width: int, hidden: int, moe: MoEConfig | None = None) -> None:
        super().__init__()
        moe = moe or MoEConfig()
        self.moe = moe
        e, eh = moe.num_experts, moe.resolve_expert_hidden(hidden)
        self.width, self.expert_hidden = width, eh
        self.router = nn.Linear(width, e, bias=False)
        self.fc1_weight = nn.Parameter(torch.empty(e, eh, width))
        self.fc1_bias = nn.Parameter(torch.zeros(e, eh))
        self.fc2_weight = nn.Parameter(torch.empty(e, width, eh))
        self.fc2_bias = nn.Parameter(torch.zeros(e, width))
        shared_hidden = moe.shared_experts * eh
        self.shared = DenseFFN(width, shared_hidden) if shared_hidden else None
        self.register_buffer("route_bias", torch.zeros(e))
        self.register_buffer("load_accum", torch.zeros(e), persistent=False)
        self.aux_loss: torch.Tensor | None = None
        self.last_load: torch.Tensor | None = None
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            nn.init.normal_(self.router.weight, std=0.02)  # small but non-zero: breaks ties
            for i in range(self.moe.num_experts):
                nn.init.xavier_uniform_(self.fc1_weight[i])
                nn.init.xavier_uniform_(self.fc2_weight[i])
            self.fc1_bias.zero_()
            self.fc2_bias.zero_()

    def route(self, flat: torch.Tensor):
        """flat: (T, width) -> (expert indices (T,k), combine weights (T,k), probs (T,E))."""
        dtype = torch.promote_types(flat.dtype, torch.float32)
        with torch.autocast(flat.device.type, enabled=False):
            logits = F.linear(flat.to(dtype), self.router.weight.to(dtype))
            probs = logits.softmax(dim=-1)
            scores = probs + self.route_bias.to(dtype) if self.moe.balance == "bias" else probs
            idx = scores.topk(self.moe.top_k, dim=-1).indices
            picked = probs.gather(-1, idx)
            weights = picked / picked.sum(dim=-1, keepdim=True)
        return idx, weights, probs

    def _expert(self, i: int, h: torch.Tensor) -> torch.Tensor:
        h = F.gelu(F.linear(h, self.fc1_weight[i], self.fc1_bias[i]), approximate="tanh")
        return F.linear(h, self.fc2_weight[i], self.fc2_bias[i])

    def forward(self, x: torch.Tensor, hw=None) -> torch.Tensor:
        b, n, w = x.shape
        flat = x.reshape(-1, w)
        idx, weights, probs = self.route(flat)
        k, num_experts = idx.shape[1], self.moe.num_experts
        assigned = idx.reshape(-1)  # assignment a = token * k + slot
        counts = torch.bincount(assigned, minlength=num_experts)
        self._record(counts, probs)

        order = torch.argsort(assigned, stable=True)  # groups assignments by expert
        tokens = order // k
        combine = weights.reshape(-1)[order]
        out = flat.new_zeros(flat.shape[0], w, dtype=weights.dtype)
        start = 0
        for i, n_i in enumerate(counts.tolist()):  # dropless: every assignment is processed
            if n_i == 0:
                continue
            t = tokens[start : start + n_i]
            y = self._expert(i, flat[t])
            out.index_add_(0, t, y.to(out.dtype) * combine[start : start + n_i, None])
            start += n_i
        out = out.to(x.dtype).view(b, n, w)
        return out if self.shared is None else out + self.shared(x)

    def _record(self, counts: torch.Tensor, probs: torch.Tensor) -> None:
        load = counts.float() / counts.sum().clamp(min=1)
        self.last_load = load.detach()
        self.aux_loss = None
        if not self.training:
            return
        if self.moe.balance == "aux_loss":
            balance = (load.detach() * probs.mean(dim=0)).sum()
            self.aux_loss = self.moe.aux_loss_weight * self.moe.num_experts * balance
        elif self.moe.balance == "bias":
            self.load_accum += counts.to(self.load_accum.dtype)


def moe_layers(model: nn.Module) -> list[tuple[str, MoEFFN]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, MoEFFN)]


def collect_aux_loss(model: nn.Module) -> torch.Tensor:
    """Sum of the weighted aux losses from the last training forward (0 if none)."""
    losses = [m.aux_loss for _, m in moe_layers(model) if m.aux_loss is not None]
    if losses:
        return torch.stack(losses).sum()
    return torch.zeros((), device=next(model.parameters()).device)


def moe_metrics(model: nn.Module) -> dict[str, float]:
    """Load-balance numbers from the last forward, ready for logging.

    ``max_over_mean`` is the busiest expert's load relative to a perfectly even split (1.0 =
    balanced); ``entropy`` is the load entropy relative to its maximum (1.0 = balanced).
    """
    out: dict[str, float] = {}
    worst = 0.0
    for name, m in moe_layers(model):
        if m.last_load is None:
            continue
        e = m.last_load.numel()
        load = m.last_load
        ent = -(load * load.clamp(min=1e-12).log()).sum().item() / math.log(e) if e > 1 else 1.0
        out[f"moe/{name}/max_over_mean"] = load.max().item() * e
        out[f"moe/{name}/min_over_mean"] = load.min().item() * e
        out[f"moe/{name}/entropy"] = ent
        worst = max(worst, out[f"moe/{name}/max_over_mean"])
    if out:
        out["moe/max_over_mean"] = worst
    return out


@torch.no_grad()
def step_router_bias(model: nn.Module) -> None:
    """Loss-free balancing update; call once per optimizer step (balance="bias" layers only)."""
    for _, m in moe_layers(model):
        if m.moe.balance != "bias" or m.load_accum.sum() == 0:
            continue
        underloaded = m.load_accum.mean() - m.load_accum
        m.route_bias += m.moe.bias_update_rate * underloaded.sign()
        m.load_accum.zero_()
