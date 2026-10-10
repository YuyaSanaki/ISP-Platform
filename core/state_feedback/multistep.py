"""Feedback policy of State-feedback ISP: feedback after every step.

State-feedback ISP feeds the model's inferred response back after every perturbation step,
including the last, so each step acts on the state reranked after the previous one:

    step 1 -> feedback -> step 2 -> feedback -> ... -> step T -> feedback (end point)

This is what makes it sequential. Without feedback between steps (the no-feedback baseline)
the result depends only on the final encoding.

The number of feedback events equals the number of steps. Every event adds a bounded
displacement and the displacement is not undone, so the error grows with the number of steps
(docs/state_feedback_decode_methods.md, "Multi-step stability evaluation").

Settings for other policies (one event at a chosen step, no event after the last step, an event
cap) were removed; ``check_feedback_config`` rejects configs that still ask for them.
"""
from __future__ import annotations

import warnings
from typing import Any, Mapping

ERROR_GROWTH_NOTE = (
    "Feedback runs after every step, so a chain of N steps has N feedback events. Each event "
    "adds a displacement that is not undone, so the error grows with the number of steps; "
    "keep chains short and compare only chains with the same number of steps."
)

# Removed key -> the only value that still describes feedback after every step.
REMOVED_FEEDBACK_KEYS: dict[str, Any] = {
    "feedback_every_step": True,
    "feedback_after_step": 1,
    "feedback_after_last_step": True,
}


def check_feedback_config(sf_cfg: Mapping[str, Any], n_steps: int) -> None:
    """Reject removed feedback settings that would change the policy; warn on the rest."""
    for key, every_step_value in REMOVED_FEEDBACK_KEYS.items():
        if key not in sf_cfg:
            continue
        value = sf_cfg[key]
        if type(every_step_value)(value) != every_step_value:
            raise ValueError(
                f"state_feedback.{key}: {value!r} is no longer supported. State-feedback ISP "
                "feeds back after every step, including the last; remove the key."
            )
        warnings.warn(
            f"state_feedback.{key} is ignored: feedback always runs after every step",
            stacklevel=2,
        )
    multi = sf_cfg.get("multi_step")
    if multi:
        cap = dict(multi).get("max_feedback_events")
        if cap is not None and int(cap) < int(n_steps):
            raise ValueError(
                f"state_feedback.multi_step.max_feedback_events={cap} is below the number of "
                f"steps ({n_steps}). The cap was removed: every step gets a feedback event; "
                "remove the multi_step block."
            )
        warnings.warn(
            "state_feedback.multi_step is ignored: the number of feedback events equals the "
            "number of steps",
            stacklevel=2,
        )
