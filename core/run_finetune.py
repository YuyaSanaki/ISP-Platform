"""
Fine-tuning driver for Geneformer cell/disease classification.

Configuration: YAML file (default /app/core/config/finetune.yaml).
Override path with --config or FINETUNE_CONFIG env var.

Outputs: {output_root}/{YYYYMMDD}/[finetune_<UTC time>/] containing
  - model checkpoint (safetensors/config.json)
  - results.csv, eval_results.json, preds.pkl, label_dict.json
  - figures/  (UMAP plots if enabled)
  - provenance metadata

Usage:
  docker compose run --rm finetune
  docker compose run --rm finetune --config /app/core/config/my_finetune.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Resolved once at import so CPU-only hosts do not hard-crash on `.to("cuda")`.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _empty_device_cache() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

sys.path.append(os.getcwd())
from geneformer.auto_batch_size import (
    coerce_batch_size,
    is_auto,
    resolve_batch_size,
    train_candidates_for_seq_len,
    train_memory_fraction_for_seq_len,
)
from geneformer.collator_for_classification import DataCollatorForCellClassification
from geneformer.training_schedule import resolve_dataloader_num_workers, resolve_warmup_steps
from geneformer.backends import resolve_pretrained_path
from geneformer.species_context import (
    backend_from_config,
    default_finetune_batch_size,
    log_species_banner,
    species_from_config,
)
from run_pipeline_log import install_rotating_stdio_tee
from run_provenance import write_service_provenance, update_service_provenance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_get(m: Mapping[str, Any] | None, *keys: str, default: Any = None) -> Any:
    cur: Any = m
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_finetune_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Fine-tune config not found: {path}. "
            "Create config/finetune.yaml or pass --config."
        )
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Fine-tune config must be a mapping at root: {path}")
    return data


def _utc_run_folder_name(now_utc: datetime) -> str:
    return f"finetune_{now_utc.strftime('%H%M%S')}_{now_utc.microsecond:06d}Z"


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    pre = precision_score(labels, preds, average="macro", zero_division=0)
    rec = recall_score(labels, preds, average="macro", zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {
        "accuracy": acc,
        "macro_precision": pre,
        "macro_recall": rec,
        "macro_f1": macro_f1,
    }


def format_finetune_run_banner(cfg: dict[str, Any], extras: dict[str, Any]) -> str:
    """Human-readable config summary for console + log."""
    paths = cfg.get("paths") or {}
    ft = cfg.get("finetune") or {}
    mdl = cfg.get("model") or {}
    tr = cfg.get("training") or {}
    umap_cfg = cfg.get("umap") or {}

    lines = [
        "=" * 72,
        "Fine-tuning run configuration",
        "=" * 72,
        f"  config file:        {extras.get('config_path', '')}",
        f"  dataset:            {paths.get('dataset')}",
        f"  geneformer_model:   {paths.get('geneformer_model')}",
        f"  output_root:        {paths.get('output_root')}",
        f"  output date:        {extras.get('date_used', '')}",
        f"  run folder:         {extras.get('run_folder', '')}",
        "",
        "  finetune:",
        f"    task_type:        {ft.get('task_type', 'disease')}",
        f"    label_column:     {ft.get('label_column', 'disease')}",
        f"    organ_key:        {ft.get('organ_key', None)}",
        f"    rare_threshold:   {ft.get('rare_class_threshold', 0.005)}",
        f"    train_split:      {ft.get('train_split', 0.8)}",
        "",
        "  model:",
        f"    max_input_size:   {mdl.get('max_input_size', 2048)}",
        f"    freeze_layers:    {mdl.get('freeze_layers', 0)}",
        "",
        "  training:",
        f"    learning_rate:    {tr.get('learning_rate', 5e-5)}",
        f"    batch_size:       {tr.get('batch_size', 6)}",
        f"    epochs:           {tr.get('epochs', 10)}",
        f"    lr_scheduler:     {tr.get('lr_scheduler_type', 'linear')}",
        f"    warmup_steps:     {tr.get('warmup_steps', 500)}",
        f"    warmup_ratio:     {tr.get('warmup_ratio', '(not set)')}",
        f"    num_runs:         {tr.get('num_runs', 1)}",
        f"    seed:             {tr.get('seed', 42)}",
        "",
        "  umap:",
        f"    enabled:          {umap_cfg.get('enabled', False)}",
        "=" * 72,
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Metadata injection
# ---------------------------------------------------------------------------

def apply_metadata(dataset, meta_cfg: dict[str, Any]):
    """Apply rename / add / derive column transformations to dataset."""
    if not meta_cfg:
        return dataset

    # 1. Rename existing columns
    renames = meta_cfg.get("rename_columns") or {}
    for old_name, new_name in renames.items():
        if old_name in dataset.column_names:
            print(f"  Renaming column: {old_name} → {new_name}")
            dataset = dataset.rename_column(old_name, new_name)
        else:
            print(f"  WARNING: Column '{old_name}' not found for rename, skipping.")

    # 2. Add fixed-value columns
    adds = meta_cfg.get("add_columns") or {}
    for col_name, value in adds.items():
        if col_name not in dataset.column_names:
            print(f"  Adding column: {col_name} = '{value}'")
            dataset = dataset.map(
                lambda example, v=value, c=col_name: {c: v},
                num_proc=1,
            )
        else:
            print(f"  Column '{col_name}' already exists, skipping add.")

    # 3. Derive columns from existing ones using a mapping
    derives = meta_cfg.get("derive_columns") or {}
    for col_name, derive_cfg in derives.items():
        source = derive_cfg.get("source")
        mapping = derive_cfg.get("mapping", {})
        if source and source in dataset.column_names:
            print(f"  Deriving column: {col_name} from {source} ({len(mapping)} mappings)")

            def _derive(example, s=source, m=mapping, c=col_name):
                return {c: m.get(example[s], example[s])}

            dataset = dataset.map(_derive, num_proc=1)
        else:
            print(f"  WARNING: Source column '{source}' not found for derive, skipping.")

    return dataset


# ---------------------------------------------------------------------------
# Truncation & padding
# ---------------------------------------------------------------------------

def truncate_and_pad(dataset, max_len: int, pad_id: int):
    """Truncate input_ids to max_len and create attention_mask."""

    def _process(batch):
        new_ids, new_masks, new_lens = [], [], []
        has_mask = "attention_mask" in batch
        for idx, ids in enumerate(batch["input_ids"]):
            # Truncate
            trunc = ids[:max_len]
            length = min(
                batch.get("length", [len(ids)])[idx],
                max_len,
            )
            # Pad input_ids
            if len(trunc) < max_len:
                trunc = trunc + [pad_id] * (max_len - len(trunc))
            new_ids.append(trunc)
            # Attention mask
            if has_mask:
                mask = batch["attention_mask"][idx][:max_len]
                if len(mask) < max_len:
                    mask = mask + [0] * (max_len - len(mask))
            else:
                mask = [1] * length + [0] * (max_len - length)
            new_masks.append(mask)
            new_lens.append(length)

        batch["input_ids"] = new_ids
        batch["attention_mask"] = new_masks
        batch["length"] = new_lens
        return batch

    return dataset.map(_process, batched=True)


def resolve_train_batch_size(
    value: Any,
    model,
    *,
    max_len: int,
    pad_id: int,
    fp16: bool,
    model_dir: str,
    num_labels: int,
    default: int = 6,
    require_idle: bool = False,
    skip_cache: bool = False,
) -> int:
    """Turn training batch_size="auto" into a number measured on this GPU.

    Unlike inference, batch size changes optimization dynamics, so "auto" is
    opt-in and the chosen value is printed and stored with the run.
    """
    if not is_auto(value):
        return int(value)

    device = next(model.parameters()).device
    base_ids = torch.full((max_len,), pad_id, dtype=torch.long)
    candidates = train_candidates_for_seq_len(max_len)
    memory_fraction = train_memory_fraction_for_seq_len(max_len)
    # AdamW keeps two fp32 moment buffers per trainable param; the bare
    # forward+backward probe omits them and underestimates real Trainer memory.
    # Critical for long-seq human V2 (Mouse→Human): probe can pass batch=6 then
    # OOM on the first training step.
    opt_buffers: list[torch.Tensor] = []

    def _ensure_optimizer_like_buffers() -> None:
        if opt_buffers:
            return
        for p in model.parameters():
            if not p.requires_grad:
                continue
            opt_buffers.append(torch.zeros_like(p, dtype=torch.float32))
            opt_buffers.append(torch.zeros_like(p, dtype=torch.float32))

    def probe(batch_size: int) -> None:
        _ensure_optimizer_like_buffers()
        input_ids = base_ids.unsqueeze(0).expand(batch_size, max_len).contiguous().to(device)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        model.train()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=bool(fp16) and device.type == "cuda",
        ):
            loss = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            ).loss
        loss.backward()
        model.zero_grad(set_to_none=True)

    return resolve_batch_size(
        value,
        default=default,
        probe=probe,
        cache_fields={
            "task": "finetune_train",
            "model": model_dir,
            "seq_len": max_len,
            "num_labels": num_labels,
            "fp16": bool(fp16),
            "train_memory_fraction": memory_fraction,
            "optimizer_like_buffers": True,
        },
        candidates=candidates,
        memory_fraction=memory_fraction,
        label="training batch_size",
        require_idle=require_idle,
        skip_cache=skip_cache,
    )


# ---------------------------------------------------------------------------
# UMAP generation
# ---------------------------------------------------------------------------

def generate_umap(
    model_path: str,
    eval_ds,
    data_collator,
    inv_label_dict: dict,
    group_name: str,
    figures_dir: Path,
    batch_size: int,
    seed: int,
    umap_cfg: dict[str, Any],
):
    """Generate PCA → UMAP projection plot and save as PDF."""
    try:
        import umap
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from torch.utils.data import DataLoader
    except ImportError as e:
        print(f"  UMAP skipped — missing dependency: {e}")
        return

    print(f"  Generating UMAP for {group_name}...")

    umap_model = BertForSequenceClassification.from_pretrained(
        model_path, output_hidden_states=True
    ).to(DEVICE)
    umap_model.eval()

    loader = DataLoader(
        eval_ds,
        batch_size=max(1, batch_size // 2),
        collate_fn=data_collator,
    )

    embs, lbls = [], []
    with torch.no_grad():
        for b in loader:
            inp = {
                k: v.to(DEVICE)
                for k, v in b.items()
                if k in ["input_ids", "attention_mask"]
            }
            out = umap_model(**inp)
            hs = out.hidden_states[-1]
            cls = hs[:, 0, :].half().cpu().numpy()
            embs.append(cls)
            if "labels" in b:
                lbls.extend(b["labels"].cpu().numpy())

    del umap_model
    _empty_device_cache()

    if not embs:
        print("  UMAP skipped — no embeddings extracted.")
        return

    embs = np.vstack(embs).astype(np.float32)
    lbls = np.array(lbls)

    n_samp, n_feat = embs.shape
    n_pca = min(umap_cfg.get("pca_components", 50), n_samp, n_feat)
    red = PCA(n_components=n_pca, random_state=seed).fit_transform(embs)
    proj = umap.UMAP(
        n_components=2,
        random_state=seed,
        low_memory=True,
        n_neighbors=umap_cfg.get("n_neighbors", 15),
    ).fit_transform(red)

    # Plot
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(
        proj[:, 0], proj[:, 1], c=lbls, cmap="tab10", s=10, alpha=0.7
    )
    unique = np.unique(lbls)
    handles = []
    for u in unique:
        handles.append(
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=scatter.cmap(scatter.norm(u)),
                label=inv_label_dict.get(int(u), str(u)),
            )
        )
    plt.legend(handles=handles, title="Class", loc="best")
    plt.title(f"{group_name} — Fine-tuned Embedding UMAP")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    pdf_path = figures_dir / f"{group_name}_umap.pdf"
    plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  UMAP saved: {pdf_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="To copy construct from a tensor",
        category=UserWarning,
    )

    default_cfg = os.environ.get("FINETUNE_CONFIG", "/app/core/config/finetune.yaml")
    p = argparse.ArgumentParser(
        description="Fine-tune Geneformer for cell/disease classification."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path(default_cfg),
        help="YAML config path (default: FINETUNE_CONFIG or /app/core/config/finetune.yaml).",
    )
    args = p.parse_args()

    cfg = load_finetune_config(args.config)

    # ── Paths ──
    paths = cfg.get("paths") or {}
    species = species_from_config(cfg)
    backend = backend_from_config(cfg)
    # L1: keep YAML max_input_size aligned with the selected model backend
    # (actual truncate uses checkpoint max_position_embeddings; this avoids
    # misleading 2048 banners / future readers when running Human V2 standalone).
    cfg.setdefault("model", {})
    yaml_mis = cfg["model"].get("max_input_size")
    cfg["model"]["max_input_size"] = int(backend.max_input_size)
    if yaml_mis is not None and int(yaml_mis) != int(backend.max_input_size):
        print(
            f"  NOTE: model.max_input_size {yaml_mis} → {backend.max_input_size} "
            f"(from species backend {species.get('model')})"
        )
    log_species_banner(species, prefix="  ")
    dataset_path = paths.get("dataset")
    model_dir = paths.get("geneformer_model") or resolve_pretrained_path(species, paths)
    output_root = paths.get("output_root")
    if not dataset_path or not model_dir or not output_root:
        raise ValueError(
            "paths.dataset, paths.geneformer_model, and paths.output_root are all required."
        )

    # ── Output directory ──
    date_used = datetime.now().strftime("%Y%m%d")
    explicit_run_dir = paths.get("run_dir")
    if explicit_run_dir:
        run_root = Path(str(explicit_run_dir))
        run_folder = "(paths.run_dir)"
    else:
        run_root = Path(output_root) / date_used
        want_time_subdir = _deep_get(cfg, "paths", "output_time_subdir", default=True)
        if want_time_subdir:
            run_folder = _utc_run_folder_name(datetime.now(timezone.utc))
            run_root = run_root / run_folder
        else:
            run_folder = "(flat under date)"

    run_root.mkdir(parents=True, exist_ok=True)

    # ── Logging ──
    log_path = run_root / "finetune_run.log"
    install_rotating_stdio_tee(log_path, env_prefix="FINETUNE")

    print(
        format_finetune_run_banner(
            cfg,
            extras={
                "config_path": str(args.config.resolve()),
                "date_used": date_used,
                "run_folder": run_folder,
            },
        ),
        end="",
    )
    print(f"  run log (rotating): {log_path}")

    # ── CUDA check ──
    print("Checking CUDA...", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))

    # ── Provenance ──
    prov_cfg = cfg.get("provenance") or {}
    enable_fp = prov_cfg.get("enable_input_fingerprint", False)
    if os.environ.get("FINETUNE_FINGERPRINT_INPUTS"):
        enable_fp = os.environ["FINETUNE_FINGERPRINT_INPUTS"].lower() in ("1", "true", "yes")
    is_fast = prov_cfg.get("fingerprint_fast", True)

    if enable_fp:
        os.environ["ENABLE_INPUT_FINGERPRINT"] = "true"

    write_service_provenance(
        run_root=run_root,
        service="finetune",
        config_path=args.config.resolve(),
        extra_meta={
            "service": "finetune",
            "dataset": dataset_path,
            "geneformer_model": model_dir,
        },
        input_paths={
            "dataset": dataset_path,
            "geneformer_model": model_dir,
        },
        fast_fingerprint=is_fast,
    )

    # ── Load dataset ──
    print(f"\nLoading dataset: {dataset_path}")
    full_dataset = load_from_disk(dataset_path)
    print(f"  Columns: {full_dataset.column_names}")
    print(f"  Rows: {len(full_dataset)}")

    # ── Metadata injection ──
    meta_cfg = cfg.get("metadata") or {}
    if meta_cfg:
        print("\nApplying metadata transformations...")
        full_dataset = apply_metadata(full_dataset, meta_cfg)
        print(f"  Columns after metadata: {full_dataset.column_names}")

    # ── Config sections ──
    ft_cfg = cfg.get("finetune") or {}
    mdl_cfg = cfg.get("model") or {}
    tr_cfg = cfg.get("training") or {}
    umap_cfg = cfg.get("umap") or {}
    rt_cfg = cfg.get("runtime") or {}

    task_type = ft_cfg.get("task_type", "disease")
    label_column = ft_cfg.get("label_column", task_type)
    organ_key = ft_cfg.get("organ_key")
    rare_threshold = float(ft_cfg.get("rare_class_threshold", 0.005))
    train_split = float(ft_cfg.get("train_split", 0.8))
    shuffle_seed = int(ft_cfg.get("shuffle_seed", 42))

    freeze_layers = int(mdl_cfg.get("freeze_layers", 0))
    ignore_mismatched = mdl_cfg.get("ignore_mismatched_sizes", True)

    lr = float(tr_cfg.get("learning_rate", 5e-5))
    default_bs = default_finetune_batch_size(int(backend.max_input_size))
    batch_size = coerce_batch_size(tr_cfg.get("batch_size", default_bs), default=default_bs)
    if "batch_size" not in tr_cfg and default_bs != 6:
        print(
            f"  NOTE: training.batch_size default → {default_bs} "
            f"(from backend max_input_size {backend.max_input_size})"
        )
    elif (
        not is_auto(batch_size)
        and int(batch_size) > default_bs
        and int(backend.max_input_size) > 2048
    ):
        print(
            f"  WARNING: training.batch_size={batch_size} exceeds the safe default "
            f"({default_bs}) for max_input_size={backend.max_input_size}. "
            "Long-sequence fine-tuning often OOMs at batch sizes calibrated for 2048. "
            f"Consider batch_size={default_bs} or gradient accumulation."
        )
    eval_batch_size_cfg = tr_cfg.get("eval_batch_size")
    lr_sched = tr_cfg.get("lr_scheduler_type", "linear")
    epochs = int(tr_cfg.get("epochs", 10))
    weight_decay = float(tr_cfg.get("weight_decay", 0.001))
    fp16 = tr_cfg.get("fp16", True)
    base_seed = int(tr_cfg.get("seed", 42))
    num_runs = int(tr_cfg.get("num_runs", 1))
    nproc = int(rt_cfg.get("nproc", 16))

    do_umap = umap_cfg.get("enabled", False)
    figures_dir = run_root / "figures"

    # ── Validate label column exists ──
    if label_column not in full_dataset.column_names:
        raise ValueError(
            f"Label column '{label_column}' not found in dataset. "
            f"Available columns: {full_dataset.column_names}. "
            f"Use metadata.rename_columns or metadata.add_columns to fix this."
        )

    # ── Build groups ──
    if organ_key and organ_key in full_dataset.column_names:
        group_names = sorted(set(full_dataset[organ_key]))
        print(f"\nGrouping by '{organ_key}': {group_names}")
    else:
        if organ_key:
            print(
                f"\nWARNING: organ_key='{organ_key}' not in columns. "
                "Training on full dataset."
            )
        group_names = ["all"]

    pipeline_ok = False
    saved_model_paths = []

    try:
        for group_name in group_names:
            print(f"\n{'='*60}")
            print(f"Processing group: {group_name}")
            print(f"{'='*60}")

            # Filter for this group
            if group_name == "all":
                group_ds = full_dataset
            else:
                group_ds = full_dataset.filter(
                    lambda ex, gn=group_name, ok=organ_key: ex[ok] == gn,
                    num_proc=nproc,
                )

            # Drop rare classes
            class_counter = Counter(group_ds[label_column])
            total_cells = sum(class_counter.values())
            classes_to_keep = [
                k for k, v in class_counter.items()
                if v > (rare_threshold * total_cells)
            ]
            dropped = set(class_counter.keys()) - set(classes_to_keep)
            if dropped:
                print(f"  Dropping rare classes (<{rare_threshold*100:.1f}%): {dropped}")
                group_ds = group_ds.filter(
                    lambda ex, ctk=classes_to_keep, lc=label_column: ex[lc] in ctk,
                    num_proc=nproc,
                )

            # Shuffle and rename label column
            group_ds = group_ds.shuffle(seed=shuffle_seed)
            if label_column != "label":
                group_ds = group_ds.rename_column(label_column, "label")

            # Remove organ_key column if it exists and is not label
            if organ_key and organ_key in group_ds.column_names and organ_key != "label":
                group_ds = group_ds.remove_columns(organ_key)

            # Create label dictionary
            target_names = sorted(set(group_ds["label"]))
            label_dict = {name: i for i, name in enumerate(target_names)}
            inv_label_dict = {i: name for name, i in label_dict.items()}
            print(f"  Label dict: {label_dict}")
            print(f"  Total cells: {len(group_ds)}")

            # Map labels to numeric IDs
            def _classes_to_ids(example, ld=label_dict):
                example["label"] = ld[example["label"]]
                return example

            group_ds = group_ds.map(_classes_to_ids, num_proc=nproc)

            # Train/eval split
            n_train = round(len(group_ds) * train_split)
            train_ds = group_ds.select(range(0, n_train))
            eval_ds = group_ds.select(range(n_train, len(group_ds)))

            # Filter eval for labels that appear in train
            trained_labels = set(train_ds["label"])
            eval_ds = eval_ds.filter(
                lambda ex, tl=trained_labels: ex["label"] in tl,
                num_proc=nproc,
            )
            print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)}")

            # ── Run loop ──
            accuracy_list, f1_list = [], []
            best_acc, best_run, best_df = -np.inf, None, None

            for run_idx in range(num_runs):
                seed = base_seed + run_idx
                print(f"\n--- Run {run_idx+1}/{num_runs} (seed={seed}) ---")
                set_seed(seed)

                # Load model
                model = BertForSequenceClassification.from_pretrained(
                    model_dir,
                    num_labels=len(label_dict),
                    output_attentions=False,
                    output_hidden_states=False,
                    ignore_mismatched_sizes=ignore_mismatched,
                ).to(DEVICE)

                # Freeze layers
                if freeze_layers > 0:
                    modules_to_freeze = model.bert.encoder.layer[:freeze_layers]
                    for module in modules_to_freeze:
                        for param in module.parameters():
                            param.requires_grad = False
                    print(f"  Froze {freeze_layers} bottom layers")

                max_len = model.config.max_position_embeddings
                pad_id = model.config.pad_token_id or 0

                # Truncate and pad
                train_proc = truncate_and_pad(train_ds, max_len, pad_id)
                eval_proc = truncate_and_pad(eval_ds, max_len, pad_id)

                train_proc.set_format(
                    "torch",
                    columns=["input_ids", "attention_mask", "label", "length"],
                )
                eval_proc.set_format(
                    "torch",
                    columns=["input_ids", "attention_mask", "label", "length"],
                )

                run_batch_size = batch_size
                if is_auto(run_batch_size):
                    run_batch_size = resolve_train_batch_size(
                        run_batch_size,
                        model,
                        max_len=max_len,
                        pad_id=pad_id,
                        fp16=fp16,
                        model_dir=str(model_dir),
                        num_labels=len(label_dict),
                        default=default_bs,
                    )
                    set_seed(seed)  # calibration consumed RNG draws (dropout)
                eval_batch_size = (
                    int(eval_batch_size_cfg)
                    if eval_batch_size_cfg is not None
                    else max(1, run_batch_size // 4)
                )

                steps_per_epoch = max(1, len(train_proc) // run_batch_size)
                total_steps = steps_per_epoch * epochs
                warmup = resolve_warmup_steps(
                    tr_cfg, steps_per_epoch=steps_per_epoch, num_epochs=epochs
                )
                dataloader_workers = resolve_dataloader_num_workers(rt_cfg)
                print(
                    f"  Training steps: {total_steps} "
                    f"({steps_per_epoch}/epoch × {epochs} epochs), "
                    f"warmup_steps={warmup}, "
                    f"dataloader_num_workers={dataloader_workers}"
                )

                # Output directory
                out_dir = str(run_root / f"{group_name}_run{run_idx+1}")
                os.makedirs(out_dir, exist_ok=True)

                # Training arguments
                log_steps = max(1, len(train_proc) // run_batch_size // 10)
                training_args = TrainingArguments(
                    output_dir=out_dir,
                    learning_rate=lr,
                    seed=seed,
                    do_train=True,
                    do_eval=True,
                    eval_strategy="epoch",
                    save_strategy="epoch",
                    logging_steps=log_steps,
                    fp16=fp16,
                    dataloader_num_workers=dataloader_workers,
                    dataloader_pin_memory=True,
                    group_by_length=True,
                    length_column_name="length",
                    disable_tqdm=False,
                    lr_scheduler_type=lr_sched,
                    warmup_steps=warmup,
                    weight_decay=weight_decay,
                    per_device_train_batch_size=run_batch_size,
                    per_device_eval_batch_size=eval_batch_size,
                    num_train_epochs=epochs,
                    load_best_model_at_end=True,
                    eval_accumulation_steps=1,
                )

                data_collator = DataCollatorForCellClassification()

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=train_proc,
                    eval_dataset=eval_proc,
                    compute_metrics=compute_metrics,
                )

                # Train
                trainer.train()
                _empty_device_cache()

                # Predict
                preds = trainer.predict(eval_proc)
                _empty_device_cache()

                # Metrics
                acc = preds.metrics.get("test_accuracy")
                f1 = preds.metrics.get("test_macro_f1")
                if acc is not None:
                    accuracy_list.append(acc)
                if f1 is not None:
                    f1_list.append(f1)

                # Track best
                if acc is not None and acc > best_acc:
                    best_acc, best_run = acc, run_idx + 1
                    pred_ids = np.argmax(preds.predictions, axis=1)
                    best_df = pd.DataFrame(
                        {
                            "True_Label": [
                                inv_label_dict.get(int(i), "Unknown")
                                for i in preds.label_ids
                            ],
                            "Pred_Label": [
                                inv_label_dict.get(int(i), "Unknown")
                                for i in pred_ids
                            ],
                            "Logits": [l.tolist() for l in preds.predictions],
                        }
                    )

                print(
                    f"  acc={acc*100:.1f}% | macro_f1={f1*100:.1f}% "
                    f"(best run={best_run})"
                    if acc is not None and f1 is not None
                    else f"  acc={acc} | macro_f1={f1}"
                )

                # Save model and metrics
                trainer.save_model(out_dir)
                trainer.save_metrics("eval", preds.metrics)

                with open(os.path.join(out_dir, "preds.pkl"), "wb") as fp:
                    pickle.dump(preds, fp)

                pd.DataFrame(
                    {
                        "True": [
                            inv_label_dict.get(int(i), "Unknown")
                            for i in preds.label_ids
                        ],
                        "Pred": [
                            inv_label_dict.get(int(i), "Unknown")
                            for i in np.argmax(preds.predictions, axis=1)
                        ],
                    }
                ).to_csv(os.path.join(out_dir, "results.csv"), index=False)

                # Save label dictionary
                with open(os.path.join(out_dir, "label_dict.json"), "w") as fp:
                    json.dump(label_dict, fp, indent=2)

                saved_model_paths.append(out_dir)

                # UMAP on last run
                if do_umap and run_idx == num_runs - 1:
                    del model
                    _empty_device_cache()
                    generate_umap(
                        model_path=out_dir,
                        eval_ds=eval_proc,
                        data_collator=data_collator,
                        inv_label_dict=inv_label_dict,
                        group_name=group_name,
                        figures_dir=figures_dir,
                        batch_size=run_batch_size,
                        seed=seed,
                        umap_cfg=umap_cfg,
                    )

            # Summary for this group
            if accuracy_list:
                m, s = np.mean(accuracy_list) * 100, np.std(accuracy_list) * 100
                print(f"\n=== {group_name} Accuracy ===")
                print(f"  Mean: {m:.1f}%, Std: {s:.1f}%")
            if best_df is not None:
                best_dir = str(run_root / f"best_{group_name}")
                os.makedirs(best_dir, exist_ok=True)
                best_df.to_csv(
                    os.path.join(best_dir, "best_results.csv"), index=False
                )
                print(f"  Best results CSV: {best_dir}")

        pipeline_ok = True

        # ── Final summary ──
        print("\n" + "=" * 72)
        print("Fine-tuning complete!")
        print("=" * 72)
        print("\nSaved model checkpoints:")
        for mp in saved_model_paths:
            print(f"  → {mp}")
        print(
            "\nTo use a fine-tuned model with ISP, update your isp.yaml:"
        )
        print(f"  paths.geneformer_model: {saved_model_paths[-1] if saved_model_paths else '???'}")
        print("  model.type: CellClassifier")
        print(f"  model.num_classes: {len(label_dict)}")

    finally:
        update_service_provenance(
            run_root,
            {
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "run_status": "completed" if pipeline_ok else "failed",
                "saved_model_paths": [str(p) for p in saved_model_paths],
            },
            service="finetune",
        )


if __name__ == "__main__":
    main()
