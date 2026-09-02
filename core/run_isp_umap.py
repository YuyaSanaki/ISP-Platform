"""
ISP UMAP: per-cell embedding trajectories for targeted gene perturbations.

Supports single- or multi-gene **group** perturbations (delete / overexpress), aligned with
Fig.3 OSKM4-style in-silico OE/KD. Standalone: docker compose run --rm isp_umap.
Pipeline / run_isp.py: invoked automatically when perturbation.genes_to_perturb is non-empty.
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from umap_plot_style import plot_isp_umap_scatter
import yaml
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_UMAP_CFG: dict[str, Any] = {
    "enabled": True,
    "seed": 42,
    "n_neighbors": 15,
    "min_dist": 0.1,
    "pca_components": 0,
    "show_trajectory_arrows": True,
    "num_trajectory_arrows": 100,
    "max_cells_per_state": 2000,
    "batch_size": 100,
}

# Locked Fig.2/3/4 manuscript UMAP preprocessing (Paper_idea_v2 §8.2).
FIG_UMAP_PCA_COMPONENTS = 50
FIG_UMAP_SEED = 0


def _validate_local_model_path(model_path: str | Path) -> Path:
    path = Path(model_path)
    if not path.is_dir():
        raise FileNotFoundError(
            f"Fine-tuned model directory not found: {path}. "
            "Use --run-dir with a completed pipeline folder, or set paths.geneformer_model."
        )
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Missing config.json in model directory: {path}")
    return path


def load_isp_cfg_from_pipeline_run(run_dir: Path) -> dict[str, Any]:
    """Load ISP stage config written by run_pipeline.py under stage_configs/isp.yaml."""
    run_dir = run_dir.expanduser().resolve()
    isp_cfg_path = run_dir / "stage_configs" / "isp.yaml"
    if not isp_cfg_path.is_file():
        raise FileNotFoundError(
            f"Missing {isp_cfg_path}. Run the pipeline ISP stage first, or pass --config with explicit paths."
        )
    with open(isp_cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid ISP config at {isp_cfg_path}")
    cfg.setdefault("umap", dict(DEFAULT_UMAP_CFG))
    cfg.setdefault("runtime", {})
    cfg["runtime"].setdefault("num_classes", (cfg.get("model") or {}).get("num_classes", 2))
    return cfg


def _normalize_genes_list(genes: str | list[str] | None) -> list[str]:
    if genes is None:
        return []
    if isinstance(genes, str):
        genes = [genes]
    out: list[str] = []
    seen: set[str] = set()
    for gene in genes:
        g = str(gene).strip()
        if g and g not in seen:
            out.append(g)
            seen.add(g)
    return out


def perturbation_genes_from_cfg(pert: Mapping[str, Any]) -> list[str]:
    """Extract gene list from a perturbation block (genes_to_perturb or legacy gene_to_perturb)."""
    genes = _normalize_genes_list(pert.get("genes_to_perturb"))
    if not genes:
        genes = _normalize_genes_list(pert.get("gene_to_perturb"))
    return genes


def perturbation_display_label(
    genes: list[str],
    gene_labels: Mapping[str, str] | None = None,
    *,
    fallback: str | None = None,
) -> str:
    if fallback:
        return fallback
    if gene_labels:
        return "+".join(gene_labels.get(g, g) for g in genes)
    return "+".join(genes)


def build_isp_umap_config(
    isp_cfg: Mapping[str, Any],
    genes: str | list[str],
    *,
    gene_label: str | None = None,
    gene_labels: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build an ISP UMAP config from a run_isp / pipeline ISP stage config."""
    pert = isp_cfg.get("perturbation") or {}
    paths = isp_cfg.get("paths") or {}
    runtime = isp_cfg.get("runtime") or {}
    umap_overrides = dict(isp_cfg.get("umap") or {})
    batch_size = umap_overrides.pop("batch_size", runtime.get("batch_size", DEFAULT_UMAP_CFG["batch_size"]))

    umap_cfg = {**DEFAULT_UMAP_CFG, **umap_overrides}
    gene_list = _normalize_genes_list(genes)
    label = perturbation_display_label(gene_list, gene_labels, fallback=gene_label)

    return {
        "paths": {
            "dataset": paths["dataset"],
            "geneformer_model": paths["geneformer_model"],
        },
        "species": dict(isp_cfg.get("species") or {}),
        "perturbation": {
            "genes_to_perturb": gene_list,
            "gene_label": label,
            "type": pert.get("type", "delete"),
            "state_key": pert.get("state_key", "disease"),
            "start_state": pert["start_state"],
            "end_state": pert["end_state"],
        },
        "runtime": {
            "num_classes": int((isp_cfg.get("model") or {}).get("num_classes", 2)),
            "batch_size": int(batch_size),
        },
        "umap": umap_cfg,
    }


def resolve_perturbation_tokens(
    cfg: Mapping[str, Any],
    genes: list[str],
) -> tuple[list[int], list[str]]:
    """Resolve gene symbols/Ensembl IDs to model token IDs."""
    from geneformer.gene_converter import normalize_gene_id, resolve_gene_for_model
    from geneformer.species_context import backend_from_config, load_input_symbol_table, species_from_config

    if not genes:
        raise ValueError("At least one perturbation gene is required.")

    species = species_from_config(cfg)
    backend = backend_from_config(cfg)
    symbol_table = load_input_symbol_table(species)

    with open(backend.token_dictionary, "rb") as f:
        token_dict = pickle.load(f)

    tokens: list[int] = []
    resolved_ids: list[str] = []
    for raw_gene in genes:
        normalized = normalize_gene_id(raw_gene)
        if normalized in token_dict:
            ensembl_id = normalized
        else:
            ensembl_id = resolve_gene_for_model(
                raw_gene,
                species,
                input_symbol_table=symbol_table,
            )
        if not ensembl_id:
            raise ValueError(
                f"Could not resolve gene '{raw_gene}' for species "
                f"model_organism={species.get('model_organism')} model={species.get('model')}"
            )
        gene_token = token_dict.get(ensembl_id)
        if gene_token is None:
            raise ValueError(
                f"Token for {ensembl_id} ({raw_gene}) not found in model token dictionary."
            )
        if ensembl_id != raw_gene:
            logger.info("Resolved %s -> %s (token %s)", raw_gene, ensembl_id, gene_token)
        tokens.append(int(gene_token))
        resolved_ids.append(ensembl_id)
    return tokens, resolved_ids


def apply_group_delete(example: Mapping[str, Any], tokens_to_perturb: list[int]) -> dict[str, Any]:
    from geneformer.in_silico_perturber import delete_indices

    ex = dict(example)
    ids = list(ex["input_ids"])
    ex["input_ids"] = ids
    token_set = set(tokens_to_perturb)
    indices = [i for i, tok in enumerate(ids) if tok in token_set]
    ex["perturb_index"] = indices if indices else [-100]
    if ex["perturb_index"] != [-100]:
        ex = delete_indices(ex)
    length = len(ex["input_ids"])
    ex["length"] = length
    ex["attention_mask"] = [1] * length
    return ex


def apply_group_overexpress(example: Mapping[str, Any], tokens_to_perturb: list[int]) -> dict[str, Any]:
    from geneformer.in_silico_perturber import overexpress_tokens

    ex = dict(example)
    ids = list(ex["input_ids"])
    ex["input_ids"] = ids
    ex["tokens_to_perturb"] = tokens_to_perturb
    present = [ids.index(t) for t in tokens_to_perturb if t in ids]
    ex["perturb_index"] = present if present else [-100]
    ex = overexpress_tokens(ex)
    length = len(ex["input_ids"])
    ex["length"] = length
    ex["attention_mask"] = [1] * length
    return ex


def perturb_dataset_group(
    dataset,
    tokens_to_perturb: list[int],
    perturb_type: str,
    batch_size: int,
):
    """Apply a group delete or overexpress perturbation to every row in a HF dataset."""
    perturb_type = str(perturb_type or "delete").lower()
    if perturb_type not in {"delete", "overexpress"}:
        raise ValueError(
            f"ISP UMAP supports perturbation.type delete or overexpress; got {perturb_type!r}."
        )
    apply_fn = apply_group_delete if perturb_type == "delete" else apply_group_overexpress

    def _map_batch(batch):
        out_ids, out_len, out_mask = [], [], []
        for i in range(len(batch["input_ids"])):
            row = {k: batch[k][i] for k in batch}
            pert = apply_fn(row, tokens_to_perturb)
            out_ids.append(pert["input_ids"])
            out_len.append(pert["length"])
            out_mask.append(pert["attention_mask"])
        return {"input_ids": out_ids, "length": out_len, "attention_mask": out_mask}

    cols = [c for c in dataset.column_names if c not in {"input_ids", "length", "attention_mask"}]
    meta = {c: dataset[c] for c in cols}
    perturbed = dataset.map(
        _map_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names,
    )
    for col, values in meta.items():
        perturbed = perturbed.add_column(col, values)
    return perturbed


def should_run_isp_umap(genes_to_perturb: list[str], cfg: Mapping[str, Any]) -> bool:
    """Run UMAP only for targeted genes (not genome-wide 'all')."""
    if not genes_to_perturb:
        return False
    umap_cfg = cfg.get("umap") or {}
    return bool(umap_cfg.get("enabled", True))


def _effective_n_neighbors(n_neighbors: int, n_samples: int) -> int:
    return min(int(n_neighbors), max(2, int(n_samples) - 1))


def fit_umap_projection(all_embs: np.ndarray, umap_cfg: Mapping[str, Any]) -> tuple[np.ndarray, str]:
    """Project embeddings to 2D UMAP coordinates.

    When ``pca_components`` > 0, applies PCA then umap-learn (Fig.2/3/4-aligned).
    Otherwise fits UMAP directly on embeddings (cuML when available).
    """
    n_neighbors = int(umap_cfg.get("n_neighbors", DEFAULT_UMAP_CFG["n_neighbors"]))
    min_dist = float(umap_cfg.get("min_dist", DEFAULT_UMAP_CFG["min_dist"]))
    umap_seed = int(umap_cfg.get("seed", DEFAULT_UMAP_CFG["seed"]))
    pca_components = int(umap_cfg.get("pca_components", 0) or 0)

    input_embs = all_embs
    method = "direct UMAP"
    if pca_components > 0:
        from sklearn.decomposition import PCA

        n_pca = min(pca_components, input_embs.shape[0], input_embs.shape[1])
        logger.info(
            "PCA(%d) before UMAP on %d points × %d dims (seed %d)",
            n_pca,
            input_embs.shape[0],
            input_embs.shape[1],
            umap_seed,
        )
        input_embs = PCA(n_components=n_pca, random_state=umap_seed).fit_transform(input_embs)
        method = f"PCA({n_pca})→UMAP"

    n_neighbors = _effective_n_neighbors(n_neighbors, input_embs.shape[0])

    if pca_components > 0:
        import umap as umap_learn

        logger.info("Running umap-learn UMAP (%s, seed %d)...", method, umap_seed)
        reducer = umap_learn.UMAP(
            n_components=2,
            random_state=umap_seed,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            low_memory=True,
        )
        return reducer.fit_transform(input_embs), method

    try:
        import cuml

        logger.info("Running RAPIDS cuML GPU UMAP (%s, seed %d)...", method, umap_seed)
        reducer = cuml.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=umap_seed)
        return reducer.fit_transform(input_embs), method
    except Exception as exc:
        logger.warning("RAPIDS cuML failed: %s. Falling back to CPU UMAP...", exc)
        import umap as umap_learn

        reducer = umap_learn.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=umap_seed,
            low_memory=True,
        )
        return reducer.fit_transform(input_embs), method


def umap_position_method_label(umap_cfg: Mapping[str, Any]) -> str:
    pca_components = int(umap_cfg.get("pca_components", 0) or 0)
    seed = int(umap_cfg.get("seed", DEFAULT_UMAP_CFG["seed"]))
    if pca_components > 0:
        return f"PCA({pca_components})→UMAP (seed {seed})"
    return f"direct UMAP (seed {seed})"


def compute_mean_embs(hidden_state, length, max_len):
    """Mean pool hidden states based on actual non-padded sequence lengths."""
    device = hidden_state.device
    mask = torch.arange(max_len, device=device).unsqueeze(0) < length.unsqueeze(1)
    mask = mask.unsqueeze(-1).expand_as(hidden_state).float()
    masked_embs = hidden_state * mask
    mean_embs = masked_embs.sum(1) / length.view(-1, 1).float()
    return mean_embs


def extract_embeddings(model, dataset, device, batch_size, pad_token_id):
    """Run model forward passes to extract sequence embeddings."""
    all_embs = []
    model.eval()

    for i in range(0, len(dataset), batch_size):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        max_len = max(batch["length"])
        padded_input_ids = []
        padded_attention_masks = []

        for input_id_list in batch["input_ids"]:
            length = len(input_id_list)
            padded_input_ids.append(input_id_list + [pad_token_id] * (max_len - length))
            padded_attention_masks.append([1] * length + [0] * (max_len - length))

        input_ids = torch.tensor(padded_input_ids).to(device)
        attention_mask = torch.tensor(padded_attention_masks).to(device)
        lengths = torch.tensor(batch["length"]).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            embs = compute_mean_embs(hidden_states, lengths, max_len=input_ids.shape[1])
            all_embs.append(embs.cpu().numpy())

    if not all_embs:
        return np.array([])
    return np.concatenate(all_embs, axis=0)


_SKIP_METADATA_COLS = frozenset({"input_ids", "length", "attention_mask"})


def compute_per_cell_shifts(start_embs, pert_embs, end_embs):
    """L2 shift in embedding space and reduction in distance to the end-state centroid."""
    shift_l2 = np.linalg.norm(pert_embs - start_embs, axis=1)
    end_centroid = end_embs.mean(axis=0)
    dist_before = np.linalg.norm(start_embs - end_centroid, axis=1)
    dist_after = np.linalg.norm(pert_embs - end_centroid, axis=1)
    shift_toward_end = dist_before - dist_after
    return shift_l2, shift_toward_end


def build_per_cell_shift_table(
    start_dataset,
    start_embs,
    pert_embs,
    end_embs,
    end_state,
    umap_coords=None,
    start_umap_offset=0,
):
    """One row per perturbed start-state cell with shift metrics (and optional UMAP coords)."""
    shift_l2, shift_toward_end = compute_per_cell_shifts(start_embs, pert_embs, end_embs)
    toward_col = f"shift_toward_{end_state}"

    meta = {
        c: start_dataset[c]
        for c in start_dataset.column_names
        if c not in _SKIP_METADATA_COLS
    }
    meta["cell_index"] = list(range(len(start_dataset)))
    meta["shift_l2"] = shift_l2
    meta[toward_col] = shift_toward_end

    if umap_coords is not None:
        n = len(start_embs)
        before = umap_coords[start_umap_offset : start_umap_offset + n]
        after = umap_coords[start_umap_offset + n : start_umap_offset + 2 * n]
        meta["umap1_before"] = before[:, 0]
        meta["umap2_before"] = before[:, 1]
        meta["umap1_after"] = after[:, 0]
        meta["umap2_after"] = after[:, 1]
        meta["umap_shift_l2"] = np.linalg.norm(after - before, axis=1)

    return pd.DataFrame(meta)


def _effective_umap_batch_size(cfg: Mapping[str, Any], backend) -> int:
    """Cap dataset map / forward batch size for long human V2 sequences."""
    requested = int(cfg["runtime"].get("batch_size", DEFAULT_UMAP_CFG["batch_size"]))
    max_input = int(getattr(backend, "max_input_size", 2048))
    if max_input >= 4096:
        return min(requested, 2)
    if max_input >= 2048:
        return min(requested, 8)
    return requested


def run_isp_umap(cfg: Mapping[str, Any], output_dir: Path | str) -> Path:
    """
    Run ISP UMAP for one targeted gene set (group delete or overexpress).
    Writes PNG + per_cell_isp_shift.csv under output_dir.
    Returns the output directory used.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("ISP UMAP using device: %s", device)

    geneformer_model_path = cfg["paths"]["geneformer_model"]
    num_classes = cfg["runtime"].get("num_classes", 2)
    model_path = _validate_local_model_path(geneformer_model_path)
    logger.info("Loading sequence classification model: %s", model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        num_labels=num_classes,
        output_hidden_states=True,
        local_files_only=True,
    ).to(device)

    from geneformer.species_context import (
        backend_from_config,
        log_species_banner,
        species_from_config,
    )

    species = species_from_config(cfg)
    backend = backend_from_config(cfg)
    log_species_banner(species, prefix="  ")
    batch_size = _effective_umap_batch_size(cfg, backend)
    if batch_size != int(cfg["runtime"].get("batch_size", DEFAULT_UMAP_CFG["batch_size"])):
        logger.info("Capped UMAP forward batch_size to %d for max_input_size=%d", batch_size, backend.max_input_size)

    with open(backend.token_dictionary, "rb") as f:
        token_dict = pickle.load(f)
    pad_token_id = token_dict.get("<pad>", 0)

    pert = cfg["perturbation"]
    genes = perturbation_genes_from_cfg(pert)
    if not genes:
        raise ValueError("perturbation.genes_to_perturb (or gene_to_perturb) is required.")
    gene_label = pert.get("gene_label") or perturbation_display_label(genes)
    gene_tokens, _resolved_ids = resolve_perturbation_tokens(cfg, genes)
    logger.info(
        "Group %s perturbation for %d gene(s): %s (tokens=%s)",
        pert.get("type", "delete"),
        len(genes),
        gene_label,
        gene_tokens,
    )

    dataset_path = cfg["paths"]["dataset"]
    logger.info("Loading dataset from: %s", dataset_path)
    dataset = load_from_disk(dataset_path)

    state_key = pert["state_key"]
    start_state = pert["start_state"]
    end_state = pert["end_state"]
    umap_cfg = cfg.get("umap") or {}
    max_cells = int(umap_cfg.get("max_cells_per_state", DEFAULT_UMAP_CFG["max_cells_per_state"]))

    logger.info(
        "Filtering up to %d '%s' and '%s' cells...",
        max_cells,
        start_state,
        end_state,
    )
    start_dataset = dataset.filter(lambda x: x.get(state_key) == start_state)
    end_dataset = dataset.filter(lambda x: x.get(state_key) == end_state)

    start_dataset = start_dataset.select(range(min(len(start_dataset), max_cells)))
    end_dataset = end_dataset.select(range(min(len(end_dataset), max_cells)))
    logger.info(
        "Using %d %s cells and %d %s cells.",
        len(start_dataset),
        start_state,
        len(end_dataset),
        end_state,
    )

    perturb_type = str(pert.get("type", "delete")).lower()
    pert_start_dataset = perturb_dataset_group(
        start_dataset,
        gene_tokens,
        perturb_type,
        batch_size,
    )

    logger.info("Extracting embeddings...")
    end_embs = extract_embeddings(model, end_dataset, device, batch_size, pad_token_id)
    start_embs = extract_embeddings(model, start_dataset, device, batch_size, pad_token_id)
    pert_start_embs = extract_embeddings(model, pert_start_dataset, device, batch_size, pad_token_id)

    safe_label = str(gene_label).replace("/", "_").replace("+", "_")
    np.save(str(out_dir / f"{end_state}_embs.npy"), end_embs)
    np.save(str(out_dir / f"{start_state}_embs.npy"), start_embs)
    np.save(str(out_dir / f"{start_state}_ISP_{safe_label}_embs.npy"), pert_start_embs)

    all_embs = np.vstack([end_embs, start_embs, pert_start_embs])
    pert_label = f"{start_state}+ISP({gene_label})"
    umap_seed = int(umap_cfg.get("seed", DEFAULT_UMAP_CFG["seed"]))
    umap_embs, umap_method = fit_umap_projection(all_embs, umap_cfg)

    per_cell_df = build_per_cell_shift_table(
        start_dataset,
        start_embs,
        pert_start_embs,
        end_embs,
        end_state,
        umap_coords=umap_embs,
        start_umap_offset=len(end_embs),
    )
    per_cell_path = out_dir / "per_cell_isp_shift.csv"
    per_cell_df.to_csv(per_cell_path, index=False)
    logger.info("Saved per-cell shift metrics to %s (%d cells)", per_cell_path, len(per_cell_df))

    num_arrows = int(umap_cfg.get("num_trajectory_arrows", DEFAULT_UMAP_CFG["num_trajectory_arrows"]))
    show_arrows = bool(umap_cfg.get("show_trajectory_arrows", True)) and num_arrows > 0
    if not show_arrows:
        logger.info("Trajectory arrows disabled (show_trajectory_arrows=false or num_trajectory_arrows<=0)")

    safe_gene = safe_label
    out_file = out_dir / f"umap_{start_state}_vs_{end_state}_isp_{safe_gene}.png"
    plot_isp_umap_scatter(
        umap_embs,
        n_end=len(end_embs),
        n_start=len(start_embs),
        end_state=end_state,
        start_state=start_state,
        pert_label=pert_label,
        title=(
            f"UMAP of Embeddings: {end_state} vs {start_state} ({gene_label} ISP)\n"
            f"{umap_method}, seed {umap_seed}"
        ),
        out_path=out_file,
        show_arrows=show_arrows,
        num_arrows=num_arrows,
    )
    logger.info("Saved UMAP plot to %s", out_file)
    return out_dir


def run_isp_umap_for_targeted_genes(
    isp_cfg: Mapping[str, Any],
    genes_resolved: list[str],
    *,
    gene_labels: Mapping[str, str] | None = None,
    output_root: Path | str,
    per_gene: bool = False,
) -> list[Path]:
    """Run ISP UMAP for targeted genes under output_root.

    Default: one **group** UMAP (all genes perturbed together, Fig.3-style).
    Set ``per_gene=True`` for legacy one-subdir-per-gene runs.
    """
    output_root = Path(output_root)
    gene_labels = gene_labels or {}
    out_dirs: list[Path] = []

    if per_gene and len(genes_resolved) > 1:
        for gene in genes_resolved:
            label = gene_labels.get(gene, gene)
            gene_dir = output_root / label
            umap_cfg = build_isp_umap_config(
                isp_cfg,
                [gene],
                gene_label=label,
                gene_labels=gene_labels,
            )
            logger.info("ISP UMAP (per-gene) for %s (%s) -> %s", label, gene, gene_dir)
            out_dirs.append(run_isp_umap(umap_cfg, gene_dir))
        return out_dirs

    group_label = perturbation_display_label(genes_resolved, gene_labels)
    umap_cfg = build_isp_umap_config(
        isp_cfg,
        genes_resolved,
        gene_label=group_label,
        gene_labels=gene_labels,
    )
    logger.info(
        "ISP UMAP (group %s) for %d gene(s): %s -> %s",
        umap_cfg["perturbation"].get("type", "delete"),
        len(genes_resolved),
        group_label,
        output_root,
    )
    out_dirs.append(run_isp_umap(umap_cfg, output_root))
    return out_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="ISP UMAP Plotter")
    default_cfg = os.environ.get("ISP_UMAP_CONFIG", "/app/core/config/isp_umap.yaml")
    parser.add_argument(
        "--config",
        type=str,
        default=default_cfg,
        help="YAML config path (default: /app/core/config/isp_umap.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: /app/output/<date>/isp_umap_<time>/ or <run-dir>/isp_umap/)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Completed pipeline run directory (loads stage_configs/isp.yaml for dataset/model/genes).",
    )
    parser.add_argument(
        "--per-gene",
        action="store_true",
        help=(
            "When multiple --gene values are given, run separate UMAPs per gene "
            "(legacy). Default is one group perturbation (Fig.3-style)."
        ),
    )
    parser.add_argument(
        "--gene",
        nargs="+",
        default=None,
        metavar="GENE",
        help=(
            "Gene symbol(s) or Ensembl ID(s) to plot. Overrides genes_to_perturb from the "
            "pipeline ISP config. Required when that list is empty (genome-wide ISP)."
        ),
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=None,
        metavar="N",
        help=(
            "PCA components before UMAP (0 = direct UMAP on embeddings, default). "
            f"Use {FIG_UMAP_PCA_COMPONENTS} with --umap-seed {FIG_UMAP_SEED} for Fig.2/3/4-style plots."
        ),
    )
    parser.add_argument(
        "--umap-seed",
        type=int,
        default=None,
        metavar="SEED",
        help="UMAP (and PCA) random seed override.",
    )
    parser.add_argument(
        "--show-trajectory-arrows",
        dest="show_trajectory_arrows",
        action="store_true",
        default=None,
        help="Draw grey Start→Perturbed arrows on the UMAP (default: config / true).",
    )
    parser.add_argument(
        "--no-trajectory-arrows",
        dest="show_trajectory_arrows",
        action="store_false",
        help="Omit trajectory arrows (scatter only).",
    )
    parser.add_argument(
        "--num-trajectory-arrows",
        type=int,
        default=None,
        metavar="N",
        help="Approximate number of trajectory arrows when enabled (default: config / 100).",
    )
    args = parser.parse_args()

    def _apply_umap_cli_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
        umap = dict(cfg.get("umap") or {})
        if args.show_trajectory_arrows is not None:
            umap["show_trajectory_arrows"] = bool(args.show_trajectory_arrows)
        if args.num_trajectory_arrows is not None:
            umap["num_trajectory_arrows"] = int(args.num_trajectory_arrows)
            if int(args.num_trajectory_arrows) <= 0:
                umap["show_trajectory_arrows"] = False
        if args.pca_components is not None:
            umap["pca_components"] = max(0, int(args.pca_components))
        if args.umap_seed is not None:
            umap["seed"] = int(args.umap_seed)
        cfg = dict(cfg)
        cfg["umap"] = umap
        return cfg

    if args.run_dir is not None:
        from geneformer.species_context import (
            convert_perturbation_genes,
            load_input_symbol_table,
            species_from_config,
        )

        isp_cfg = _apply_umap_cli_overrides(load_isp_cfg_from_pipeline_run(args.run_dir))
        pert_genes = list(args.gene or [])
        if not pert_genes:
            pert_genes = list((isp_cfg.get("perturbation") or {}).get("genes_to_perturb") or [])
        if not pert_genes:
            raise ValueError(
                "No genes to plot. Pass --gene SYMBOL (or Ensembl ID), or use a pipeline "
                "ISP config with non-empty perturbation.genes_to_perturb."
            )
        species = species_from_config(isp_cfg)
        symbol_table = load_input_symbol_table(species)
        resolved, conv = convert_perturbation_genes(
            pert_genes,
            species,
            input_symbol_table=symbol_table,
        )
        if not resolved:
            raise ValueError("No perturbation genes remained after species/ortholog conversion.")
        gene_labels = {target: source for source, target in conv.mapped.items()}
        out_root = args.output_dir or (args.run_dir.expanduser().resolve() / "isp_umap")
        run_isp_umap_for_targeted_genes(
            isp_cfg,
            resolved,
            gene_labels=gene_labels,
            output_root=out_root,
            per_gene=bool(args.per_gene),
        )
        logger.info("ISP UMAP pipeline finished successfully. Output: %s", out_root)
        return

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        paths = cfg.get("paths") or {}
        explicit = paths.get("output_root")
        if explicit:
            now = datetime.now(timezone.utc)
            out_dir = Path(explicit) / now.strftime("%Y%m%d") / f"isp_umap_{now.strftime('%H%M%S')}"
        else:
            now = datetime.now(timezone.utc)
            out_dir = Path("/app/output") / now.strftime("%Y%m%d") / f"isp_umap_{now.strftime('%H%M%S')}"

    pert = cfg.get("perturbation") or {}
    genes = perturbation_genes_from_cfg(pert)
    if not genes:
        raise ValueError(
            "perturbation.genes_to_perturb (or legacy gene_to_perturb) is required for standalone ISP UMAP."
        )

    # Normalize standalone isp_umap.yaml into the shared config shape.
    if "gene_label" not in pert:
        cfg = dict(cfg)
        cfg["perturbation"] = dict(pert)
        cfg["perturbation"]["gene_label"] = perturbation_display_label(genes)
    if "genes_to_perturb" not in pert:
        cfg = dict(cfg)
        cfg["perturbation"] = dict(cfg.get("perturbation") or pert)
        cfg["perturbation"]["genes_to_perturb"] = genes
    if "umap" not in cfg:
        cfg = dict(cfg)
        cfg["umap"] = DEFAULT_UMAP_CFG.copy()
    if "species" not in cfg:
        cfg = dict(cfg)
        cfg["species"] = {"model_organism": "mouse", "model": "mouse_geneformer"}

    cfg = _apply_umap_cli_overrides(cfg)
    run_isp_umap(cfg, out_dir)
    logger.info("ISP UMAP pipeline finished successfully. Output: %s", out_dir)


if __name__ == "__main__":
    main()
