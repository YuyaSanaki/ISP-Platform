"""Residual Delta-rank decoder (Phase 2 primary method).

Maps the perturbation-induced change in contextual gene representations to a
bounded, gene-specific rank displacement:

    delta_rank = max_shift * tanh(W [delta_h ; base_rank_norm] + b)
    priority   = base_rank_norm + delta_rank

Only ``delta_h = h_pert - h_ctrl`` enters (not ``h_ctrl`` itself), so the
decoder is constrained to read the perturbation effect rather than base cell
identity. The projection is zero-initialized, so an untrained decoder is exactly
the identity permutation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn as nn


class DeltaRankDecoder(nn.Module):
    """Linear residual rank decoder over ``[delta_h ; base_rank_norm]``."""

    def __init__(self, d_model: int, max_shift: float = 0.1):
        super().__init__()
        self.d_model = int(d_model)
        self.max_shift = float(max_shift)
        self.proj = nn.Linear(self.d_model + 1, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def delta_rank(
        self,
        delta_h: torch.Tensor,
        base_rank_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Bounded rank displacement for ``[N, d]`` deltas and ``[N]`` base ranks."""
        x = torch.cat([delta_h, base_rank_norm.unsqueeze(-1)], dim=-1)
        return self.max_shift * torch.tanh(self.proj(x).squeeze(-1))

    def forward(
        self,
        delta_h: torch.Tensor,
        base_rank_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Priority (lower = leftmost)."""
        return base_rank_norm + self.delta_rank(delta_h, base_rank_norm)

    def from_states(
        self,
        h_pert: torch.Tensor,
        h_ctrl: torch.Tensor,
        base_rank_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience wrapper matching the design-doc signature."""
        return self.forward(h_pert - h_ctrl, base_rank_norm)


@dataclass
class TrainingSet:
    """Flat ``(gene, cell)`` samples for decoder training."""

    delta_h: torch.Tensor  # [N, d] float32 cpu
    base_rank: torch.Tensor  # [N]
    target: torch.Tensor  # [N] observed delta rank
    tokens: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.delta_h.size(0))

    def subset(self, mask: torch.Tensor) -> "TrainingSet":
        idx = mask.nonzero(as_tuple=True)[0]
        return TrainingSet(
            delta_h=self.delta_h.index_select(0, idx),
            base_rank=self.base_rank.index_select(0, idx),
            target=self.target.index_select(0, idx),
            tokens=[self.tokens[int(i)] for i in idx] if self.tokens else [],
        )


def token_mask(tokens: Sequence[int], keep: set[int]) -> torch.Tensor:
    return torch.tensor([int(t) in keep for t in tokens], dtype=torch.bool)


def _pairwise_loss(pred: torch.Tensor, target: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Logistic ordering loss on random pairs within a minibatch.

    Applied to *displacement*, never to the absolute priority: ranking the
    absolute priority is trivially satisfied by amplifying ``base_rank_norm``
    through the tanh, which spends the whole displacement budget on an ordering
    that is already correct by construction and destroys the delta_h signal.
    """
    n = pred.size(0)
    if n < 2:
        return pred.sum() * 0.0
    perm = torch.randperm(n, generator=generator, device=pred.device)
    a, b = pred, pred.index_select(0, perm)
    ta, tb = target, target.index_select(0, perm)
    sign = torch.sign(tb - ta)
    keep = sign != 0
    if not bool(keep.any()):
        return pred.sum() * 0.0
    diff = (b - a)[keep] * sign[keep]
    return torch.nn.functional.softplus(-diff).mean()


def train_delta_rank_decoder(
    data: TrainingSet,
    *,
    d_model: int,
    max_shift: float = 0.1,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 4096,
    lam_huber: float = 1.0,
    lam_pair: float = 0.5,
    lam_identity: float = 1.0,
    lam_smooth: float = 0.01,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> tuple[DeltaRankDecoder, dict[str, Any]]:
    """Fit the decoder; returns ``(decoder, history)``.

    Loss terms follow the design doc: Huber on displacement, a pairwise ordering
    term, an identity term (zero perturbation -> zero displacement) and a
    smoothness penalty against extreme displacements.
    """
    dev = torch.device(device) if device is not None else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    decoder = DeltaRankDecoder(d_model, max_shift=max_shift).to(dev)
    if len(data) == 0:
        return decoder, {"epochs": 0, "n_samples": 0, "final_loss": float("nan")}

    opt = torch.optim.Adam(decoder.parameters(), lr=float(lr))
    huber = nn.HuberLoss(delta=0.05)
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))
    cpu_gen = torch.Generator()
    cpu_gen.manual_seed(int(seed))

    n = len(data)
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        order = torch.randperm(n, generator=cpu_gen)
        totals = {"loss": 0.0, "huber": 0.0, "pair": 0.0, "identity": 0.0, "smooth": 0.0}
        n_batches = 0
        for start in range(0, n, int(batch_size)):
            idx = order[start : start + int(batch_size)]
            dh = data.delta_h.index_select(0, idx).to(dev)
            br = data.base_rank.index_select(0, idx).to(dev)
            tg = data.target.index_select(0, idx).to(dev)

            pred_delta = decoder.delta_rank(dh, br)
            l_huber = huber(pred_delta, tg)
            l_pair = _pairwise_loss(pred_delta, tg, gen)
            zero_delta = decoder.delta_rank(torch.zeros_like(dh), br)
            l_identity = (zero_delta**2).mean()
            l_smooth = (pred_delta**2).mean()
            loss = (
                lam_huber * l_huber
                + lam_pair * l_pair
                + lam_identity * l_identity
                + lam_smooth * l_smooth
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            totals["loss"] += float(loss.detach())
            totals["huber"] += float(l_huber.detach())
            totals["pair"] += float(l_pair.detach())
            totals["identity"] += float(l_identity.detach())
            totals["smooth"] += float(l_smooth.detach())
            n_batches += 1
        history.append({k: v / max(1, n_batches) for k, v in totals.items()} | {"epoch": epoch + 1})

    decoder.eval()
    return decoder, {
        "epochs": int(epochs),
        "n_samples": n,
        "max_shift": float(max_shift),
        "final_loss": history[-1]["loss"] if history else float("nan"),
        "history": history,
    }


@torch.no_grad()
def predict_delta_rank(
    decoder: DeltaRankDecoder,
    delta_h: torch.Tensor,
    base_rank: torch.Tensor,
) -> torch.Tensor:
    """Predicted displacement for a batch of samples (no grad)."""
    dev = next(decoder.parameters()).device
    return decoder.delta_rank(delta_h.to(dev), base_rank.to(dev)).detach().cpu()


@torch.no_grad()
def null_drift(decoder: DeltaRankDecoder, n_genes: int = 2048) -> dict[str, float]:
    """Displacement the decoder produces when ``delta_h = 0``.

    This is the null-perturbation guardrail: with no perturbation evidence the
    decoder must not move genes. Non-zero drift here means the bias / base-rank
    weight learned a spurious permutation.
    """
    dev = next(decoder.parameters()).device
    base = torch.linspace(0.0, 1.0, int(n_genes), device=dev)
    zeros = torch.zeros((int(n_genes), decoder.d_model), device=dev)
    drift = decoder.delta_rank(zeros, base).abs()
    return {
        "null_drift_mean_abs": float(drift.mean()),
        "null_drift_max_abs": float(drift.max()),
        "null_drift_rank_units_mean": float(drift.mean() * (int(n_genes) - 1)),
    }


def save_decoder(decoder: DeltaRankDecoder, path) -> None:
    torch.save(
        {
            "state_dict": decoder.state_dict(),
            "d_model": decoder.d_model,
            "max_shift": decoder.max_shift,
        },
        str(path),
    )


def load_decoder(path, device: str | torch.device | None = None) -> DeltaRankDecoder:
    blob = torch.load(str(path), map_location="cpu")
    decoder = DeltaRankDecoder(int(blob["d_model"]), max_shift=float(blob["max_shift"]))
    decoder.load_state_dict(blob["state_dict"])
    decoder.eval()
    if device is not None:
        decoder = decoder.to(torch.device(device))
    return decoder
