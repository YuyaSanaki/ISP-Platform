#!/usr/bin/env python3
"""Rank enrichment after ISP: gene set enrichment of the top genes of Start, Goal and ISP.

Runs when ``postprocess.enabled`` and ``postprocess.rank_enrichment`` are on
(``core/postprocess_switch.py``; Web UI "Cluster / cell-type / enrichment analysis"), or by
hand on a finished run directory. Settings: ``rank_enrichment`` (State-feedback:
``state_feedback.rank_enrichment``).

Start and Goal: pseudobulk rank per gene (``state_feedback.gene_ranks``), the mean over cells
of ``position / (length - 1)``, 1 where the gene is not in the encoding. Background: genes
detected in at least ``min_detection`` cells of Start and Goal (``background: both``) or of
either, minus the perturbed genes. Each list is the top ``top_fraction`` (default 10%) of the
background. ORA against local gene set libraries (``core/gene_set_enrichment.py``, no web
service) with the background as universe.

State-feedback ISP (stability run with rank enrichment; ``rank_enrichment/chains.json``):
  The ISP list is the top of the background by gene-level Delta-spec: the displacement of the
  configured chain after the potential-outcome rerank minus the mean displacement under the
  placebo chains, averaged over decoder seeds. Closeness to Goal (``state_feedback.closeness``)
  compares that change with Goal minus Start against leave-one-out placebo values.
  Figures: ``enrich_<library>`` (Start, Goal, configured schedule) and ``closeness``.

Normal ISP (per-gene screen; ``ispstats_results/*.csv`` with one row per perturbed gene):
  The ISP list is the top of the background by ``Shift_to_goal_end`` among genes with a
  positive shift and ``Goal_end_FDR`` below ``fdr``, as long as the Start list at most; its
  ORA universe is the background genes that ISP tested. Start and Goal positions are taken from
  the dataset in the run's ``isp_config_used.yaml`` (at most ``max_cells_per_state`` cells
  per state). Figure: ``enrich_<library>`` (Start, Goal, ISP).

Usage:
  python3 core/run_isp_rank_enrichment.py --run-dir <run dir>
  python3 core/run_isp_rank_enrichment.py --run-dir ... --top-fraction 0.05 --background either
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))

import enrichment_plot as ep  # noqa: E402
import gene_set_enrichment as gse  # noqa: E402
import postprocess_switch as pps  # noqa: E402
from platform_paths import localize  # noqa: E402
from state_feedback import closeness as cl  # noqa: E402
from state_feedback import gene_ranks as gr  # noqa: E402

DEFAULT_MAX_CELLS_PER_STATE = 3000


def _settings(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    out = dict(pps.RANK_ENRICHMENT_DEFAULTS)
    out.update({k: v for k, v in (cfg or {}).items() if v is not None and k != "enabled"})
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_positions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"token": int, "n_cells": int})


def _symbols(src: Path) -> dict[int, str]:
    genes = pd.read_csv(src / "genes.csv.gz", dtype={"token": int, "symbol": str})
    return dict(zip(genes["token"], genes["symbol"]))


def _ora_lists(
    lists: Mapping[str, Sequence[int]],
    background: Sequence[int],
    symbol: Mapping[int, str],
    libraries: Sequence[str],
    out: Path,
    fdr: float,
    universes: Mapping[str, Sequence[int]] | None = None,
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, Any], dict[str, dict[str, set[str]]]]:
    """ORA of each list; ``universes`` overrides the background for some lists."""
    library_sets = {lib: gse.load_library(lib) for lib in libraries}
    bg_symbols = [symbol.get(int(t), str(t)) for t in background]
    own_bg = {name: [symbol.get(int(t), str(t)) for t in u] for name, u in (universes or {}).items()}
    results: dict[str, dict[str, pd.DataFrame]] = {}
    lib_info: dict[str, Any] = {}
    for lib, sets in library_sets.items():
        gmt = gse.gene_sets_dir() / f"{lib}.gmt"
        lib_info[lib] = {"label": gse.library_label(lib), "sha256": _sha256(gmt),
                         "terms": len(sets), "n_fdr": {}}
        results[lib] = {}
        for name, tokens in lists.items():
            res = gse.ora([symbol.get(int(t), str(t)) for t in tokens],
                          own_bg.get(name, bg_symbols), sets)
            d = out / "enrichment" / name
            d.mkdir(parents=True, exist_ok=True)
            res.to_csv(d / f"{lib}.csv", index=False)
            results[lib][name] = res
            lib_info[lib]["n_fdr"][name] = int((res["Adjusted P-value"] < fdr).sum())
    return results, lib_info, library_sets


def _enrichment_figures(
    results: Mapping[str, Mapping[str, pd.DataFrame]],
    panels: Sequence[tuple[str, str]],
    s: Mapping[str, Any],
    out: Path,
) -> list[str]:
    """One figure per library; ``panels``: (list name, panel title)."""
    figures = [
        ep.plot_enrichment(
            [(title, res.get(name)) for name, title in panels],
            out / "figures" / f"enrich_{lib.lower()}",
            k=int(s["top_terms"]), fdr=float(s["fdr"]), suptitle=gse.library_label(lib),
        )
        for lib, res in results.items()
    ]
    return [str(p.relative_to(out)) for p in figures]


def _write_top_genes(lists: Mapping[str, Sequence[int]], scores: Mapping[str, pd.Series],
                     symbol: Mapping[int, str], out: Path) -> None:
    rows = []
    for name, tokens in lists.items():
        for i, t in enumerate(tokens, start=1):
            rows.append({"list": name, "order": i, "token": int(t),
                         "symbol": symbol.get(int(t), str(t)),
                         "score": float(scores[name].get(int(t), float("nan")))})
    pd.DataFrame(rows).to_csv(out / "top_genes.csv", index=False)


# ---------------------------------------------------------------- State-feedback ISP

def seed_positions(run_dir: Path) -> dict[int, dict[str, pd.DataFrame]]:
    """Final-encoding position sums per decoder seed and chain."""
    out: dict[int, dict[str, pd.DataFrame]] = {}
    for path in sorted(run_dir.glob("seed*/rank_enrichment_positions.csv.gz")):
        seed = int(path.parent.name.removeprefix("seed"))
        pos = _load_positions(path)
        out[seed] = {c: g.drop(columns="chain") for c, g in pos.groupby("chain")}
    if not out:
        raise FileNotFoundError(f"No seed*/rank_enrichment_positions.csv.gz under {run_dir}")
    return out


def chain_roles(meta: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Configured chain and placebo chains."""
    configured = str(meta["configured"])
    placebos = meta.get("placebos")
    if placebos is None:
        placebos = [c for c in meta["chain_tokens"] if c != configured]
    return configured, [str(c) for c in placebos]


def seed_delta_spec(
    run_dir: Path,
    start: pd.DataFrame,
    meta: Mapping[str, Any],
    positions: Mapping[int, Mapping[str, pd.DataFrame]] | None = None,
) -> dict[int, pd.DataFrame]:
    """Gene-level Delta-spec of the configured chain per decoder seed."""
    query, placebo_names = chain_roles(meta)
    chain_tokens = {c: {int(t) for t in ts} for c, ts in meta["chain_tokens"].items()}
    out: dict[int, pd.DataFrame] = {}
    for seed, by_chain in (positions or seed_positions(run_dir)).items():
        path = run_dir / f"seed{seed}" / "rank_enrichment_positions.csv.gz"
        if query not in by_chain:
            raise ValueError(f"{path}: no positions for chain {query}")
        placebos = {c: by_chain[c] for c in placebo_names if c in by_chain}
        if not placebos:
            raise ValueError(f"{path}: no placebo chains (Delta-spec needs random chains)")
        out[seed] = gr.gene_delta_spec(
            start, by_chain[query], placebos, chain_tokens, int(meta["n_start_cells"])
        )
    return out


def closeness_to_goal(
    table: pd.DataFrame,
    positions: Mapping[int, Mapping[str, pd.DataFrame]],
    start: pd.DataFrame,
    meta: Mapping[str, Any],
    library_sets: Mapping[str, Mapping[str, set[str]]],
    symbol: Mapping[int, str],
    results: Mapping[str, Mapping[str, pd.DataFrame]],
    s: Mapping[str, Any],
    out: Path,
) -> dict[str, Any]:
    """Closeness metrics of the configured chain against placebo values; writes ``closeness/``."""
    tokens = [int(t) for t in table.index]
    G = (table["goal_rank"] - table["start_rank"]).to_numpy()
    members = {lib: cl.set_members(tokens, symbol, sets) for lib, sets in library_sets.items()}
    configured, placebo_names = chain_roles(meta)
    pseudo_by_seed = []
    for by_chain in positions.values():
        keep = {c: by_chain[c] for c in [configured, *placebo_names] if c in by_chain}
        disp = cl.chain_displacements(start, keep, int(meta["n_start_cells"]), tokens)
        pseudo_by_seed.append(cl.query_and_placebo_deltas(disp, configured, meta["chain_tokens"])[1])
    pseudo = sum(pseudo_by_seed) / len(pseudo_by_seed)
    query = cl.all_metrics(table["delta_spec"].to_numpy(), G, members)
    placebo = {c: cl.all_metrics(pseudo[c].to_numpy(), G, members) for c in pseudo.columns}
    by_seed = {
        c.removeprefix("delta_spec_seed"): cl.all_metrics(table[c].to_numpy(), G, members)
        for c in table.columns if c.startswith("delta_spec_seed")
    }

    frac, fdr = float(s["top_fraction"]), float(s["fdr"])
    bg_symbols = [symbol.get(t, str(t)) for t in tokens]
    for lib, sets in library_sets.items():
        res = results[lib]
        level = f"ora:{lib}"
        prof = cl.ora_profile_metrics({"isp": res["isp"], "dspec": res["dspec"]}, res["goal"], fdr=fdr)
        query[level] = {f"{n}_{k}": v for n, m in prof.items() for k, v in m.items()}
        for c in pseudo.columns:
            lists = {
                "isp": gr.top_fraction(table["start_rank"] + pseudo[c], frac),
                "dspec": gr.top_fraction(pseudo[c], frac),
            }
            pres = {n: gse.ora([symbol.get(t, str(t)) for t in v], bg_symbols, sets)
                    for n, v in lists.items()}
            pm = cl.ora_profile_metrics(pres, res["goal"], fdr=fdr)
            placebo[c][level] = {f"{n}_{k}": v for n, m in pm.items() for k, v in m.items()}

    rows, flat = [], []
    summary: dict[str, Any] = {}
    for level, metrics in query.items():
        summary[level] = {}
        for metric, value in metrics.items():
            if metric == "n_terms":
                summary[level][metric] = int(value)
                continue
            nul = [placebo[c].get(level, {}).get(metric, float("nan")) for c in placebo]
            entry = cl.compare_to_placebos(value, nul)
            flat.append({"level": level, "metric": metric, **entry})
            if level in by_seed.get(next(iter(by_seed), ""), {}):
                entry["by_seed"] = {k: v[level][metric] for k, v in by_seed.items()}
            summary[level][metric] = entry
            rows.append({"level": level, "metric": metric, "chain": "configured", "value": value})
            rows.extend({"level": level, "metric": metric, "chain": c, "value": v}
                        for c, v in zip(placebo, nul))
    d = out / "closeness"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(d / "closeness_values.csv", index=False)
    pd.DataFrame(flat).to_csv(d / "closeness_summary.csv", index=False)
    pseudo.to_csv(d / "placebo_pseudo_delta_spec.csv.gz")
    return {"summary": summary, "query": query, "placebo": placebo}


def run_state_feedback(run_dir: Path, s: Mapping[str, Any], species: str | None,
                       out: Path) -> dict[str, Any]:
    src = run_dir / "rank_enrichment"
    meta = json.loads((src / "chains.json").read_text())
    species = species or str(meta.get("species_model") or "human_geneformer")
    symbol = _symbols(src)
    start = _load_positions(src / "start_positions.csv.gz")
    goal = _load_positions(src / "goal_positions.csv.gz")
    n_start, n_goal = int(meta["n_start_cells"]), int(meta["n_goal_cells"])
    configured, _placebos = chain_roles(meta)
    perturbed = {int(t) for t in meta["chain_tokens"][configured]}
    label = str(meta.get("configured_label") or "ISP")

    background = gr.background_tokens(
        start, goal, min_detection=int(s["min_detection"]), rule=str(s["background"]),
        exclude=perturbed,
    )
    positions = seed_positions(run_dir)
    per_seed = seed_delta_spec(run_dir, start, meta, positions)
    seeds = sorted(per_seed)
    stacked = pd.concat({k: v for k, v in per_seed.items()}, names=["seed", "token"])
    delta = stacked.groupby(level="token")[
        ["delta_configured", "delta_placebo_mean", "delta_spec"]
    ].mean()
    table = gr.rank_table(
        background, gr.pseudobulk_rank(start, n_start), gr.pseudobulk_rank(goal, n_goal), delta
    )
    for seed in seeds:
        table[f"delta_spec_seed{seed}"] = per_seed[seed]["delta_spec"].reindex(table.index).fillna(0.0)
    table.insert(0, "symbol", [symbol.get(int(t), str(t)) for t in table.index])

    frac = float(s["top_fraction"])
    scores = {
        "start": table["start_rank"], "goal": table["goal_rank"], "isp": table["isp_rank"],
        "dspec": table["delta_spec"],
    }
    tops = {name: gr.top_fraction(v, frac) for name, v in scores.items()}
    for name in tops:
        table[f"top_{name}"] = table.index.isin(tops[name])
    table.sort_values("isp_rank").to_csv(out / "gene_ranks.csv")
    _write_top_genes(tops, scores, symbol, out)
    seed_tops = {
        seed: gr.top_fraction(
            table["start_rank"] + per_seed[seed]["delta_spec"].reindex(table.index).fillna(0.0), frac
        )
        for seed in seeds
    }

    libraries = list(s["libraries"] or gse.default_libraries(species))
    results, lib_info, library_sets = _ora_lists(tops, background, symbol, libraries, out,
                                                 float(s["fdr"]))
    figures = _enrichment_figures(
        results, [("start", "Start"), ("goal", "Goal"), ("dspec", label)], s, out,
    )

    n_placebos = int(per_seed[seeds[0]]["n_placebos"].max())
    closeness = None
    if n_placebos >= 2:
        closeness = closeness_to_goal(table, positions, start, meta, library_sets,
                                      symbol, results, s, out)
        lib_labels = {lib: gse.library_label(lib) for lib in libraries}
        figures.append(str(ep.plot_closeness(
            label, closeness["query"], closeness["placebo"], lib_labels,
            out / "figures" / "closeness",
        ).relative_to(out)))
    summary = {
        "mode": "state_feedback",
        "run_dir": str(run_dir),
        "settings": dict(s),
        "species_vocabulary": "mouse" if str(species).lower().startswith("mouse") else "human",
        "n_start_cells": n_start,
        "n_goal_cells": n_goal,
        "goal_state": meta.get("goal_state"),
        "configured_label": label,
        "perturbed_genes": sorted(symbol.get(t, str(t)) for t in perturbed),
        "n_background": len(background),
        "n_top": {k: len(v) for k, v in tops.items()},
        "decoder_seeds": seeds,
        "n_placebo_chains": n_placebos,
        "jaccard": {
            "start_goal": gr.jaccard(tops["start"], tops["goal"]),
            "isp_goal": gr.jaccard(tops["isp"], tops["goal"]),
            "isp_start": gr.jaccard(tops["isp"], tops["start"]),
            "dspec_goal": gr.jaccard(tops["dspec"], tops["goal"]),
            "dspec_start": gr.jaccard(tops["dspec"], tops["start"]),
        },
        "isp_top_seed_jaccard_vs_mean": {
            str(k): gr.jaccard(v, tops["isp"]) for k, v in seed_tops.items()
        },
        "libraries": lib_info,
        "closeness": closeness["summary"] if closeness else None,
        "figures": figures,
    }
    print(
        f"Rank enrichment: background {len(background)} genes, top {len(tops['isp'])} per list; "
        f"Jaccard Δspec-Goal {summary['jaccard']['dspec_goal']:.3f}, "
        f"Start-Goal {summary['jaccard']['start_goal']:.3f}",
        flush=True,
    )
    if closeness:
        p = closeness["summary"]["gene"]["progress"]
        print(f"  closeness: gene progress {p['value'] * 100:.2f}% "
              f"(placebo {p['placebo_mean'] * 100:+.2f} ± {p['placebo_sd'] * 100:.2f}, "
              f"z {p['z']:.1f})", flush=True)
    return summary


# ---------------------------------------------------------------- normal ISP

def vocabulary_symbols(cfg: Mapping[str, Any]) -> dict[int, str]:
    """Token -> gene symbol (Ensembl ID where the backend has no symbol table)."""
    from geneformer.species_context import backend_from_config

    backend = backend_from_config(cfg)
    with open(backend.token_dictionary, "rb") as fh:
        vocab = pickle.load(fh)
    names = {int(t): str(k) for k, t in vocab.items() if not str(k).startswith("<")}
    if getattr(backend, "gene_symbol_to_ensembl", None):
        with open(backend.gene_symbol_to_ensembl, "rb") as fh:
            ens_to_symbol = {v: k for k, v in pickle.load(fh).items()}
        names = {t: ens_to_symbol.get(e, e) for t, e in names.items()}
    return names


def write_isp_positions(cfg: Mapping[str, Any], out: Path, *, max_cells: int) -> dict[str, int]:
    """Start and Goal position sums from the ISP dataset (``filter_data`` applied)."""
    from datasets import load_from_disk
    from datasets.utils.logging import disable_progress_bar
    from geneformer.species_context import backend_from_config

    disable_progress_bar()
    pert = cfg.get("perturbation") or {}
    state_key = str(pert.get("state_key", "disease"))
    states = {"start": str(pert["start_state"]), "goal": str(pert["end_state"])}
    filt = (cfg.get("isp") or {}).get("filter_data") or {}
    size = int(backend_from_config(cfg).max_input_size)
    ds = load_from_disk(str((cfg.get("paths") or {})["dataset"]))
    keep = ["input_ids", state_key, *[k for k in filt if k != state_key]]
    ds = ds.select_columns([c for c in dict.fromkeys(keep) if c in ds.column_names])
    if filt:
        allowed = {k: set(map(str, v if isinstance(v, (list, tuple)) else [v]))
                   for k, v in filt.items()}
        ds = ds.filter(lambda ex: all(str(ex[k]) in v for k, v in allowed.items()))
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, state in states.items():
        sub = ds.filter(lambda ex, st=state: str(ex[state_key]) == st)
        if len(sub) > max_cells:
            sub = sub.shuffle(seed=0).select(range(max_cells))
        pos, n = gr.position_sums([[int(t) for t in r][:size] for r in sub["input_ids"]])
        pos.to_csv(out / f"{name}_positions.csv.gz", index=False)
        counts[name] = n
    symbols = vocabulary_symbols(cfg)
    pd.DataFrame({"token": list(symbols), "symbol": list(symbols.values())}).to_csv(
        out / "genes.csv.gz", index=False
    )
    return counts


def isp_stats_table(run_dir: Path) -> pd.DataFrame | None:
    """Per-gene ISP stats (one row per perturbed gene), or None for group / per-cell output."""
    for path in sorted((run_dir / "ispstats_results").glob("*.csv")):
        df = pd.read_csv(path)
        if {"Gene", "Shift_to_goal_end"} <= set(df.columns):
            return df
    return None


def run_normal_isp(run_dir: Path, s: Mapping[str, Any], cfg: Mapping[str, Any] | None,
                   species: str | None, out: Path) -> dict[str, Any] | None:
    stats = isp_stats_table(run_dir)
    if stats is None:
        print("Rank enrichment: ISP stats have no per-gene rows (gene-set or per-cell "
              "output); needs a genome-wide or per-gene ISP screen. Skipped.", flush=True)
        return None
    if cfg is None:
        used = run_dir / "isp_config_used.yaml"
        if not used.exists():
            raise FileNotFoundError(f"{used} not found (needed for the dataset path)")
        cfg = localize(yaml.safe_load(used.read_text()))
    from geneformer.species_context import species_from_config

    species = species or str(species_from_config(cfg).get("model") or "human_geneformer")
    max_cells = int(s.get("max_cells_per_state") or DEFAULT_MAX_CELLS_PER_STATE)
    counts_path = out / "n_cells.json"
    if not counts_path.exists():
        counts_path.write_text(json.dumps(write_isp_positions(cfg, out, max_cells=max_cells)) + "\n")
    counts = json.loads(counts_path.read_text())
    n_start, n_goal = int(counts["start"]), int(counts["goal"])
    start = _load_positions(out / "start_positions.csv.gz")
    goal = _load_positions(out / "goal_positions.csv.gz")
    symbol = _symbols(out)
    pert = cfg.get("perturbation") or {}
    listed = [str(g) for g in pert.get("genes_to_perturb") or []]

    background = gr.background_tokens(
        start, goal, min_detection=int(s["min_detection"]), rule=str(s["background"]),
    )
    stats = stats.assign(Gene=pd.to_numeric(stats["Gene"], errors="coerce")).dropna(subset=["Gene"])
    stats = stats.assign(Gene=stats["Gene"].astype(int)).drop_duplicates("Gene").set_index("Gene")
    fdr = float(s["fdr"])
    hit = stats["Shift_to_goal_end"] > 0
    if "Goal_end_FDR" in stats:
        hit &= stats["Goal_end_FDR"] < fdr
    table = pd.DataFrame(index=pd.Index(background, name="token"))
    table.insert(0, "symbol", [symbol.get(int(t), str(t)) for t in table.index])
    table["start_rank"] = gr.pseudobulk_rank(start, n_start).reindex(table.index, fill_value=1.0)
    table["goal_rank"] = gr.pseudobulk_rank(goal, n_goal).reindex(table.index, fill_value=1.0)
    table["shift_to_goal"] = stats["Shift_to_goal_end"].reindex(table.index)
    if "Goal_end_FDR" in stats:
        table["goal_end_fdr"] = stats["Goal_end_FDR"].reindex(table.index)
    table["isp_hit"] = hit.reindex(table.index, fill_value=False)
    n_tested = int(table["shift_to_goal"].notna().sum())

    frac = float(s["top_fraction"])
    tops = {"start": gr.top_fraction(table["start_rank"], frac),
            "goal": gr.top_fraction(table["goal_rank"], frac)}
    n_list = len(tops["start"])
    hits = table.loc[table["isp_hit"], "shift_to_goal"].sort_values(ascending=False, kind="mergesort")
    tops["isp"] = [int(t) for t in hits.index[:n_list]]
    for name in tops:
        table[f"top_{name}"] = table.index.isin(tops[name])
    table.sort_values(["shift_to_goal"], ascending=False).to_csv(out / "gene_ranks.csv")
    scores = {"start": table["start_rank"], "goal": table["goal_rank"],
              "isp": table["shift_to_goal"]}
    _write_top_genes(tops, scores, symbol, out)

    libraries = list(s["libraries"] or gse.default_libraries(species))
    tested = [int(t) for t in table.index[table["shift_to_goal"].notna()]]
    results, lib_info, _sets = _ora_lists(tops, background, symbol, libraries, out, fdr,
                                          universes={"isp": tested})
    ptype = str(pert.get("type", "delete"))
    isp_title = f"ISP ({ptype})"
    figures = _enrichment_figures(
        results, [("start", "Start"), ("goal", "Goal"), ("isp", isp_title)], s, out,
    )
    summary = {
        "mode": "isp",
        "run_dir": str(run_dir),
        "settings": dict(s),
        "species_vocabulary": "mouse" if str(species).lower().startswith("mouse") else "human",
        "n_start_cells": n_start,
        "n_goal_cells": n_goal,
        "goal_state": pert.get("end_state"),
        "perturbation_type": ptype,
        "genes_to_perturb": listed or "all",
        "n_background": len(background),
        "n_tested_in_background": n_tested,
        "n_isp_hits_in_background": int(table["isp_hit"].sum()),
        "n_top": {k: len(v) for k, v in tops.items()},
        "jaccard": {
            "start_goal": gr.jaccard(tops["start"], tops["goal"]),
            "isp_goal": gr.jaccard(tops["isp"], tops["goal"]),
            "isp_start": gr.jaccard(tops["isp"], tops["start"]),
        },
        "libraries": lib_info,
        "figures": figures,
    }
    print(f"Rank enrichment: background {len(background)} genes ({n_tested} tested), "
          f"ISP list {len(tops['isp'])} of {int(table['isp_hit'].sum())} hits "
          f"(shift > 0, FDR < {fdr:g}), Start/Goal lists {n_list}", flush=True)
    return summary


# ---------------------------------------------------------------- entry points

def run(
    run_dir: Path,
    cfg: Mapping[str, Any] | None = None,
    *,
    species: str | None = None,
    out_dir: Path | None = None,
    isp_config: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Rank enrichment of a State-feedback stability run or a normal ISP run.

    ``cfg``: rank enrichment settings. ``isp_config``: the normal ISP config (default: the
    run's ``isp_config_used.yaml``).
    """
    s = _settings(cfg)
    run_dir = Path(run_dir)
    out = Path(out_dir) if out_dir else run_dir / "rank_enrichment"
    out.mkdir(parents=True, exist_ok=True)
    if (run_dir / "rank_enrichment" / "chains.json").exists():
        summary = run_state_feedback(run_dir, s, species, out)
    else:
        summary = run_normal_isp(run_dir, s, isp_config, species, out)
    if summary is None:
        return None
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    for lib, info in summary["libraries"].items():
        print(f"  {lib}: FDR<{s['fdr']} terms " + ", ".join(
            f"{k}={v}" for k, v in info["n_fdr"].items()), flush=True)
    print(f"Wrote {out}", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="State-feedback stability run or normal ISP run directory.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Default: <run-dir>/rank_enrichment (overwrites tables and figures).")
    parser.add_argument("--top-fraction", type=float, default=None)
    parser.add_argument("--background", choices=["both", "either"], default=None)
    parser.add_argument("--min-detection", type=int, default=None)
    parser.add_argument("--libraries", nargs="+", default=None)
    parser.add_argument("--fdr", type=float, default=None)
    parser.add_argument("--top-terms", type=int, default=None)
    parser.add_argument("--max-cells-per-state", type=int, default=None,
                        help="Normal ISP: Start/Goal cells read from the dataset (default 3000).")
    args = parser.parse_args()

    cfg: dict[str, Any] = {}
    manifest_path = args.run_dir / "run_manifest.json"
    used_path = args.run_dir / "isp_config_used.yaml"
    if manifest_path.exists():
        cfg = dict(json.loads(manifest_path.read_text()).get("rank_enrichment") or {})
    elif used_path.exists():
        cfg = pps.rank_enrichment_settings(yaml.safe_load(used_path.read_text()) or {})
    for key in ("top_fraction", "background", "min_detection", "libraries", "fdr", "top_terms",
                "max_cells_per_state"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    run(args.run_dir, cfg, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
