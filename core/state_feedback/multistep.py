"""Event cap for multi-step State-feedback ISP (``feedback_every_step: true``).

With feedback after every perturbation step, the chain alternates

    perturbation step t -> encoding pre_t -> feedback -> encoding post_t -> step t+1

and ``max_feedback_events`` bounds the number of feedback events per chain. After the
cap, the remaining steps still run, unreranked.

There is no per-cell stop. A convergence stop misread small whole-encoding changes
(large moves of a few genes, or a new perturbation yet to come) as convergence, and an
exact 2-cycle check practically never fires because a new perturbation enters between
feedback events.
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any, Mapping

REMOVED_KEYS = frozenset(
    {
        "converge_spearman",
        "converge_topk",
        "converge_jaccard",
        "converge_patience",
        "halt_on_cycle",
    }
)


@dataclass(frozen=True)
class MultiStepConfig:
    max_feedback_events: int = 5

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "MultiStepConfig":
        raw = dict(raw or {})
        removed = sorted(set(raw) & REMOVED_KEYS)
        if removed:
            warnings.warn(
                f"state_feedback.multi_step keys {removed} are ignored: the per-cell "
                "convergence and 2-cycle stops were removed",
                stacklevel=2,
            )
            for key in removed:
                raw.pop(key)
        unknown = sorted(set(raw) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown state_feedback.multi_step keys: {unknown}")
        cfg = cls(
            max_feedback_events=int(raw.get("max_feedback_events", cls.max_feedback_events)),
        )
        if cfg.max_feedback_events < 1:
            raise ValueError("state_feedback.multi_step.max_feedback_events must be >= 1")
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FeedbackGuard:
    """Feedback-event cap for one condition's chain."""

    def __init__(self, cfg: MultiStepConfig | None = None):
        self.cfg = cfg or MultiStepConfig()
        self.events = 0
        self.capped_before_step: int | None = None

    @property
    def cap_reached(self) -> bool:
        return self.events >= self.cfg.max_feedback_events

    def record_event(self) -> dict[str, float]:
        self.events += 1
        return {"guard_event": float(self.events)}

    def mark_cap(self, step: int) -> bool:
        """Record the first step skipped by the cap; True the first time only."""
        if self.capped_before_step is not None:
            return False
        self.capped_before_step = int(step)
        return True

    def summary(self) -> dict[str, Any]:
        return {
            "feedback_events": self.events,
            "max_feedback_events": self.cfg.max_feedback_events,
            "capped_before_step": self.capped_before_step,
        }
