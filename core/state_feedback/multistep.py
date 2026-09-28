"""Guardrails for multi-step State-feedback ISP (``feedback_every_step: true``).

With feedback after every perturbation step, the chain alternates

    perturbation step t -> encoding pre_t -> feedback -> encoding post_t -> step t+1

and the rerank can oscillate or keep making negligible moves. Two per-cell
guardrails stop feedback for a cell (its later steps still run, unreranked):

* **Cycle halt (guardrail 5).** If the proposed ``post_t`` equals ``post_{t-2}``
  while differing from ``post_{t-1}``, the rerank is oscillating between two
  encodings. The proposal is rejected (the cell keeps ``pre_t``) and the cell halts.
* **Convergence stop (guardrail 6).** A feedback event is "small" when it barely
  moves the encoding it was applied to: Spearman(pre_t, post_t) above
  ``converge_spearman`` or top-K Jaccard(pre_t, post_t) above ``converge_jaccard``.
  The Jaccard criterion applies only to cells with more than K genes: the gene set
  is fixed, so for shorter cells the top K is the whole set and Jaccard is always 1.
  After ``converge_patience`` consecutive small events the cell stops. Consecutive
  post-feedback encodings are not compared directly because the perturbation step
  between them moves the encoding by design.

A hard cap (``max_feedback_events``) bounds the number of feedback events per chain.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from state_feedback.metrics import is_two_cycle, topk_jaccard
from state_feedback.oracle_rerank import spearman_rank_correlation

HALT_CYCLE = "cycle"
HALT_CONVERGED = "converged"
HALT_CAP = "cap"


@dataclass(frozen=True)
class MultiStepConfig:
    max_feedback_events: int = 5
    converge_spearman: float = 0.995
    converge_topk: int = 1000
    converge_jaccard: float = 0.99
    converge_patience: int = 2
    halt_on_cycle: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "MultiStepConfig":
        raw = dict(raw or {})
        unknown = sorted(set(raw) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown state_feedback.multi_step keys: {unknown}")
        cfg = cls(
            max_feedback_events=int(raw.get("max_feedback_events", cls.max_feedback_events)),
            converge_spearman=float(raw.get("converge_spearman", cls.converge_spearman)),
            converge_topk=int(raw.get("converge_topk", cls.converge_topk)),
            converge_jaccard=float(raw.get("converge_jaccard", cls.converge_jaccard)),
            converge_patience=int(raw.get("converge_patience", cls.converge_patience)),
            halt_on_cycle=bool(raw.get("halt_on_cycle", cls.halt_on_cycle)),
        )
        if cfg.max_feedback_events < 1:
            raise ValueError("state_feedback.multi_step.max_feedback_events must be >= 1")
        if cfg.converge_patience < 1:
            raise ValueError("state_feedback.multi_step.converge_patience must be >= 1")
        if cfg.converge_topk < 1:
            raise ValueError("state_feedback.multi_step.converge_topk must be >= 1")
        for name in ("converge_spearman", "converge_jaccard"):
            value = getattr(cfg, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"state_feedback.multi_step.{name} must be in (0, 1]")
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FeedbackGuard:
    """Per-cell cycle halt, convergence stop and event cap for one condition's chain."""

    def __init__(self, n_cells: int, cfg: MultiStepConfig | None = None):
        self.cfg = cfg or MultiStepConfig()
        self.n_cells = int(n_cells)
        self.events = 0
        self.halt_reason: list[str | None] = [None] * self.n_cells
        self.halt_step: list[int | None] = [None] * self.n_cells
        self.n_applied = [0] * self.n_cells
        self._small_streak = [0] * self.n_cells
        self._post_prev: list[list[int] | None] = [None] * self.n_cells
        self._post_last: list[list[int] | None] = [None] * self.n_cells

    @property
    def n_active(self) -> int:
        return sum(1 for r in self.halt_reason if r is None)

    @property
    def cap_reached(self) -> bool:
        return self.events >= self.cfg.max_feedback_events

    @property
    def exhausted(self) -> bool:
        return self.cap_reached or self.n_active == 0

    def mark_cap(self, step: int) -> int:
        """Record the cap for cells still active when another event was due."""
        n = 0
        for i, reason in enumerate(self.halt_reason):
            if reason is None:
                self.halt_reason[i] = HALT_CAP
                self.halt_step[i] = int(step)
                n += 1
        return n

    def apply(
        self,
        step: int,
        before: Sequence[Sequence[int]],
        proposed: Sequence[Sequence[int]],
    ) -> tuple[list[list[int]], dict[str, float]]:
        """Filter one feedback event; returns the encodings to use and event counts."""
        if len(before) != self.n_cells or len(proposed) != self.n_cells:
            raise ValueError(
                f"FeedbackGuard expects {self.n_cells} cells, got "
                f"{len(before)} before / {len(proposed)} proposed"
            )
        self.events += 1
        cfg = self.cfg
        applied: list[list[int]] = []
        n_active_before = self.n_active
        new_cycle = 0
        new_converged = 0
        for i, (pre, prop) in enumerate(zip(before, proposed)):
            pre = [int(t) for t in pre]
            if self.halt_reason[i] is not None:
                applied.append(pre)
                continue
            prop = [int(t) for t in prop]
            if cfg.halt_on_cycle and is_two_cycle(self._post_prev[i], self._post_last[i] or pre, prop):
                self.halt_reason[i] = HALT_CYCLE
                self.halt_step[i] = int(step)
                new_cycle += 1
                applied.append(pre)
                continue
            applied.append(prop)
            self.n_applied[i] += 1
            self._post_prev[i] = self._post_last[i]
            self._post_last[i] = prop
            rho = spearman_rank_correlation(pre, prop)
            small = rho > cfg.converge_spearman
            if not small and len(pre) > cfg.converge_topk:
                jac = topk_jaccard(pre, prop, cfg.converge_topk)
                small = jac == jac and jac > cfg.converge_jaccard
            self._small_streak[i] = self._small_streak[i] + 1 if small else 0
            if self._small_streak[i] >= cfg.converge_patience:
                self.halt_reason[i] = HALT_CONVERGED
                self.halt_step[i] = int(step)
                new_converged += 1
        stats = {
            "guard_event": float(self.events),
            "guard_active_before": float(n_active_before),
            "guard_new_cycle_halts": float(new_cycle),
            "guard_new_converged": float(new_converged),
            "guard_active_after": float(self.n_active),
        }
        return applied, stats

    def cell_rows(self) -> list[dict[str, Any]]:
        """One row per cell: why and when feedback stopped (blank = ran to the end)."""
        return [
            {
                "cell": i,
                "halt_reason": self.halt_reason[i] or "",
                "halt_step": self.halt_step[i] if self.halt_step[i] is not None else "",
                "n_feedback_applied": self.n_applied[i],
            }
            for i in range(self.n_cells)
        ]

    def summary(self) -> dict[str, Any]:
        counts = {HALT_CYCLE: 0, HALT_CONVERGED: 0, HALT_CAP: 0}
        for reason in self.halt_reason:
            if reason in counts:
                counts[reason] += 1
        return {
            "feedback_events": self.events,
            "n_cells": self.n_cells,
            "halted_cycle": counts[HALT_CYCLE],
            "halted_converged": counts[HALT_CONVERGED],
            "halted_cap": counts[HALT_CAP],
            "ran_to_end": self.n_active,
        }
