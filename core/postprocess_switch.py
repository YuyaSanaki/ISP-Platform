"""One switch for the analyses after ISP: cluster / cell-type analysis and rank enrichment.

``postprocess.enabled`` turns both on. ``postprocess.celltype_prediction`` and
``postprocess.rank_enrichment`` (default true) turn either off on their own. Cluster and
cell-type analysis runs after ISP UMAP (``run_isp_umap.py``). Rank enrichment runs after
normal ISP (``run_isp.py``, per-gene screens) and after the State-feedback stability runner.

Configs without a ``postprocess`` block keep the older State-feedback switch
``state_feedback.rank_enrichment.enabled``.
"""
from __future__ import annotations

import argparse
from typing import Any, Mapping

RANK_ENRICHMENT_DEFAULTS: dict[str, Any] = {
    "top_fraction": 0.1,
    "background": "both",
    "min_detection": 5,
    "libraries": None,
    "fdr": 0.05,
    "top_terms": 10,
}


def _post(cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    post = cfg.get("postprocess")
    if post is True or post is False:
        return {"enabled": post}
    return dict(post) if isinstance(post, Mapping) else None


def rank_enrichment_enabled(cfg: Mapping[str, Any]) -> bool:
    post = _post(cfg)
    if post is not None and "enabled" in post:
        return bool(post["enabled"]) and bool(post.get("rank_enrichment", True))
    sf = (cfg.get("state_feedback") or {}).get("rank_enrichment") or {}
    return bool(sf.get("enabled", False))


def rank_enrichment_settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Defaults, then top-level ``rank_enrichment``, then ``state_feedback.rank_enrichment``."""
    out = dict(RANK_ENRICHMENT_DEFAULTS)
    for block in (cfg.get("rank_enrichment"),
                  (cfg.get("state_feedback") or {}).get("rank_enrichment")):
        if isinstance(block, Mapping):
            out.update({k: v for k, v in block.items() if k != "enabled" and v is not None})
    return out


def add_postprocess_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--enable-postprocess", dest="postprocess_enabled", action="store_true",
                   default=None,
                   help="Run the analyses after ISP: cluster / cell-type analysis and rank "
                        "enrichment (postprocess.enabled).")
    g.add_argument("--skip-postprocess", dest="postprocess_enabled", action="store_false",
                   help="Skip them even if postprocess.enabled is true.")
    r = parser.add_mutually_exclusive_group()
    r.add_argument("--rank-enrichment", dest="rank_enrichment", action="store_true",
                   default=None,
                   help="Include rank enrichment when postprocess runs (default).")
    r.add_argument("--no-rank-enrichment", dest="rank_enrichment", action="store_false",
                   help="Leave out rank enrichment when postprocess runs.")


def apply_postprocess_args(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    enabled = getattr(args, "postprocess_enabled", None)
    rank = getattr(args, "rank_enrichment", None)
    if enabled is None and rank is None:
        return cfg
    post = _post(cfg)
    if post is None:
        legacy = (cfg.get("state_feedback") or {}).get("rank_enrichment") or {}
        post = {"enabled": bool(legacy.get("enabled", False))}
    if enabled is not None:
        post["enabled"] = bool(enabled)
    if rank is not None:
        post["rank_enrichment"] = bool(rank)
    cfg["postprocess"] = post
    return cfg
