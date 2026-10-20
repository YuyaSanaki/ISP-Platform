"""
Shared helpers for the Tokenize → Fine-tune → ISP end-to-end pipeline.

Resolves paths from a minimal pipeline config (chiefly data.input_dir) and builds
per-stage YAML configs by merging defaults from config/*.yaml with overrides.
"""
from __future__ import annotations

import copy
import json
import re
import shutil
from pathlib import Path
from typing import Any, Mapping

import yaml

from data_input_layout import resolve_single_cell_input_dir
from geneformer.auto_batch_size import is_auto
from geneformer.backends import get_backend, parse_species_config, resolve_pretrained_path
from geneformer.species_context import default_isp_forward_batch_size, should_convert
from platform_paths import localize, localize_path

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = localize_path("/app/output")


def _batch_size_value(value: Any) -> int | str:
    """Keep "auto" (measured at runtime on the GPU); otherwise coerce to int."""
    if is_auto(value):
        return "auto"
    return int(value)


def pipeline_runtime(pipeline: Mapping[str, Any]) -> dict[str, Any]:
    """Shared runtime knobs from config/pipeline.yaml (applied to all stages)."""
    raw = pipeline.get("runtime")
    return dict(raw) if isinstance(raw, dict) else {}


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return localize(data)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge override into a copy of base (override wins)."""
    out: dict[str, Any] = copy.deepcopy(dict(base))
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def study_name_from_input_dir(input_dir: str) -> str:
    name = Path(input_dir.rstrip("/")).name
    if not name:
        raise ValueError(f"Cannot derive study name from input_dir: {input_dir!r}")
    safe = re.sub(r"[^\w.-]+", "_", name).strip("_")
    if not safe:
        raise ValueError(f"Invalid study name derived from input_dir: {input_dir!r}")
    return safe


def resolve_pipeline_paths(pipeline: Mapping[str, Any], run_dir: Path) -> dict[str, str]:
    """
    Compute all chained paths for one pipeline run.

    run_dir: e.g. /app/output/20260525/pipeline_120000_abc123/
    """
    data = pipeline.get("data") or {}
    paths_cfg = pipeline.get("paths") or {}

    input_dir = str(data.get("input_dir", "")).strip()
    if not input_dir:
        raise ValueError("pipeline data.input_dir is required.")

    study_root = resolve_single_cell_input_dir(input_dir)
    input_path = Path(study_root)
    prefix = data.get("output_prefix")
    study = str(prefix).strip() if prefix else study_name_from_input_dir(str(study_root))

    tokenized_dir = run_dir / "tokenized_dataset"
    loom_temp_dir = run_dir / "loom_files"
    # Provisional until the stages run: run_pipeline replaces them with
    # resolve_tokenized_dataset() and resolve_finetune_model_dir().
    dataset_path = tokenized_dir / f"{study}_0.dataset"
    finetune_run_dir = run_dir / "finetune"
    finetune_model_dir = finetune_run_dir / "all_run1"

    species = parse_species_config(pipeline.get("species"))
    backend = get_backend(species)
    pretrained_model = resolve_pretrained_path(species, paths_cfg)

    return {
        "input_dir": str(input_path),
        "study_name": study,
        "loom_temp_dir": str(loom_temp_dir),
        "tokenized_dir": str(tokenized_dir),
        "dataset_path": str(dataset_path),
        "pretrained_model": pretrained_model,
        "output_root": str(paths_cfg.get("output_root") or DEFAULT_OUTPUT_ROOT),
        "pipeline_run_dir": str(run_dir),
        "finetune_run_dir": str(finetune_run_dir),
        "finetune_model_dir": str(finetune_model_dir),
        "species": species,
        "max_input_size": backend.max_input_size,
    }


def _shard_entries(tokenized_dir: Path, study: str) -> list[tuple[int, Path]]:
    pattern = re.compile(rf"^{re.escape(study)}_(\d+)\.dataset$")
    shards: list[tuple[int, Path]] = []
    if tokenized_dir.is_dir():
        for path in tokenized_dir.iterdir():
            match = pattern.match(path.name)
            if match and path.is_dir():
                shards.append((int(match.group(1)), path))
    return sorted(shards)


def tokenized_shard_paths(tokenized_dir: Path, study: str) -> list[Path]:
    """Tokenizer shards ``{study}_{n}.dataset`` in shard order (the tokenizer starts a new
    shard each time the cells read so far reach ``tokenizer.max_cells``)."""
    shards = _shard_entries(tokenized_dir, study)
    indices = [i for i, _ in shards]
    if indices and indices != list(range(len(indices))):
        raise ValueError(
            f"Tokenized shards under {tokenized_dir} are not numbered 0..{len(indices) - 1}: "
            f"{[p.name for _, p in shards]}. Re-run tokenize into a fresh run directory."
        )
    return [p for _, p in shards]


def merged_dataset_path(tokenized_dir: Path, study: str) -> Path:
    return tokenized_dir / f"{study}_merged.dataset"


def remove_tokenized_outputs(tokenized_dir: Path, study: str) -> list[Path]:
    """Delete shards and the merged dataset left by an earlier tokenize in this run directory."""
    removed: list[Path] = []
    stale = [p for _, p in _shard_entries(tokenized_dir, study)]
    merged = merged_dataset_path(tokenized_dir, study)
    for path in [*stale, merged, merged.with_suffix(".json")]:
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(path)
        elif path.is_file():
            path.unlink()
            removed.append(path)
    return removed


def resolve_tokenized_dataset(tokenized_dir: Path, study: str) -> Path:
    """
    Dataset path for fine-tune and ISP covering every tokenizer shard.

    One shard is used as is. Several shards are concatenated in shard order into
    ``{study}_merged.dataset`` (reused while ``{study}_merged.json`` lists the same shards).
    """
    shards = tokenized_shard_paths(tokenized_dir, study)
    if not shards:
        raise FileNotFoundError(
            f"Tokenized dataset not found: no {study}_<n>.dataset under {tokenized_dir}. "
            "Run tokenize or fix data.output_prefix / input_dir."
        )
    if len(shards) == 1:
        return shards[0]

    merged = merged_dataset_path(tokenized_dir, study)
    manifest = merged.with_suffix(".json")
    names = [p.name for p in shards]
    if merged.is_dir() and manifest.is_file():
        with open(manifest, encoding="utf-8") as f:
            if (json.load(f) or {}).get("shards") == names:
                return merged

    from datasets import concatenate_datasets, load_from_disk

    parts = [load_from_disk(str(p)) for p in shards]
    combined = concatenate_datasets(parts)
    if merged.exists():
        shutil.rmtree(merged)
    combined.save_to_disk(str(merged))
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(
            {"shards": names, "cells_per_shard": [len(p) for p in parts], "cells": len(combined)},
            f,
            indent=2,
        )
    print(
        f"Merged {len(shards)} tokenized shards ({len(combined)} cells) into {merged}",
        flush=True,
    )
    return merged


_FINETUNE_RUN_DIR = re.compile(r"^(?P<group>.+)_run(?P<run>\d+)$")


def _eval_accuracy(run_dir: Path) -> float | None:
    path = run_dir / "eval_results.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        metrics = json.load(f) or {}
    for key in ("test_accuracy", "eval_accuracy"):
        if metrics.get(key) is not None:
            return float(metrics[key])
    return None


def resolve_finetune_model_dir(finetune_run_dir: Path) -> Path:
    """
    Fine-tuned checkpoint for ISP under ``finetune/{group}_run{k}/``.

    With ``training.num_runs > 1`` the run with the highest eval accuracy is used (ties and
    runs without accuracy fall back to the lowest run number, as in run_finetune's best_*).
    ``finetune.organ_key`` groups give one model per group, which ISP cannot choose between.
    """
    runs: dict[str, list[tuple[int, Path]]] = {}
    if finetune_run_dir.is_dir():
        for path in finetune_run_dir.iterdir():
            match = _FINETUNE_RUN_DIR.match(path.name)
            if match and path.is_dir() and (path / "label_dict.json").is_file():
                runs.setdefault(match.group("group"), []).append((int(match.group("run")), path))
    if not runs:
        raise FileNotFoundError(
            f"No fine-tuned model under {finetune_run_dir} "
            "(expected {group}_run{k}/ with label_dict.json). Check the fine-tune log."
        )
    if len(runs) > 1:
        raise ValueError(
            f"Fine-tune wrote one model per group {sorted(runs)} (finetune.organ_key is set). "
            "Pipeline ISP needs a single model: set stages.finetune.finetune.organ_key: null, "
            "or run ISP separately with paths.geneformer_model set to one group's run."
        )
    (candidates,) = runs.values()
    candidates.sort()
    best_path, best_acc = candidates[0][1], _eval_accuracy(candidates[0][1])
    for _, path in candidates[1:]:
        acc = _eval_accuracy(path)
        if acc is not None and (best_acc is None or acc > best_acc):
            best_path, best_acc = path, acc
    return best_path


def build_tokenize_config(
    pipeline: Mapping[str, Any],
    resolved: Mapping[str, str],
    template: Mapping[str, Any],
) -> dict[str, Any]:
    data = pipeline.get("data") or {}
    overrides = (pipeline.get("stages") or {}).get("tokenize") or {}
    cfg = deep_merge(template, overrides)

    cfg.setdefault("data", {})
    cfg["data"].update(
        {
            "input_type": data.get("input_type", cfg["data"].get("input_type", "single-cell")),
            "input_dir": resolved["input_dir"],
            "loom_temp_dir": resolved["loom_temp_dir"],
            "output_dir": resolved["tokenized_dir"] + "/",
            "output_prefix": resolved["study_name"],
        }
    )
    cfg["species"] = dict(resolved.get("species") or {})
    rt = pipeline_runtime(pipeline)
    cfg.setdefault("tokenizer", {})
    if "nproc" in rt:
        cfg["tokenizer"]["nproc"] = int(rt["nproc"])
    if "max_cells" in rt:
        cfg["tokenizer"]["max_cells"] = int(rt["max_cells"])
    species_cfg = cfg.get("species") or {}
    if cfg["tokenizer"].get("report_conversion") is None:
        cfg["tokenizer"]["report_conversion"] = should_convert(species_cfg)
    return cfg


def build_finetune_config(
    pipeline: Mapping[str, Any],
    resolved: Mapping[str, str],
    template: Mapping[str, Any],
) -> dict[str, Any]:
    overrides = (pipeline.get("stages") or {}).get("finetune") or {}
    cfg = deep_merge(template, overrides)

    cfg.setdefault("paths", {})
    cfg["paths"].update(
        {
            "dataset": resolved["dataset_path"],
            "geneformer_model": resolved["pretrained_model"],
            "output_root": resolved["finetune_run_dir"],
            "output_time_subdir": False,
            "run_dir": resolved["finetune_run_dir"],
        }
    )
    cfg["species"] = dict(resolved.get("species") or {})
    cfg.setdefault("model", {})
    cfg["model"]["max_input_size"] = resolved.get("max_input_size", cfg["model"].get("max_input_size", 2048))
    rt = pipeline_runtime(pipeline)
    if "nproc" in rt:
        cfg.setdefault("runtime", {})
        cfg["runtime"]["nproc"] = int(rt["nproc"])
    if "train_batch_size" in rt:
        cfg.setdefault("training", {})
        cfg["training"]["batch_size"] = _batch_size_value(rt["train_batch_size"])
    return cfg


def build_isp_config(
    pipeline: Mapping[str, Any],
    resolved: Mapping[str, str],
    template: Mapping[str, Any],
    *,
    finetune_model_dir: str,
    num_classes: int,
) -> dict[str, Any]:
    overrides = (pipeline.get("stages") or {}).get("isp") or {}
    pert_defaults = pipeline.get("perturbation") or {}
    cfg = deep_merge(template, overrides)

    cfg.setdefault("paths", {})
    cfg["paths"].update(
        {
            "dataset": resolved["dataset_path"],
            "geneformer_model": finetune_model_dir,
            "output_root": resolved["pipeline_run_dir"],
            "output_time_subdir": False,
            "output_date_subdir": False,
        }
    )
    cfg.setdefault("perturbation", {})
    for key in ("type", "organ_data", "state_key", "start_state", "end_state", "alt_states", "genes_to_perturb"):
        if key in pert_defaults and pert_defaults[key] is not None:
            cfg["perturbation"][key] = pert_defaults[key]

    cfg.setdefault("model", {})
    cfg["model"]["type"] = cfg["model"].get("type", "CellClassifier")
    cfg["model"]["num_classes"] = num_classes
    cfg["species"] = dict(resolved.get("species") or {})
    cfg["model"]["max_input_size"] = resolved.get("max_input_size", cfg["model"].get("max_input_size", 2048))
    rt = pipeline_runtime(pipeline)
    if rt:
        cfg.setdefault("runtime", {})
        if "nproc" in rt:
            cfg["runtime"]["nproc"] = int(rt["nproc"])
        if "forward_batch_size" in rt:
            cfg["runtime"]["forward_batch_size"] = _batch_size_value(rt["forward_batch_size"])
        elif "forward_batch_size" not in cfg.get("runtime", {}):
            mis = int(resolved.get("max_input_size", 2048))
            cfg["runtime"]["forward_batch_size"] = default_isp_forward_batch_size(mis)
    return cfg


def write_yaml(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(dict(data), f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def format_pipeline_banner(
    pipeline_path: Path,
    resolved: Mapping[str, str],
    stage_config_paths: Mapping[str, Path],
) -> str:
    lines = [
        "=" * 72,
        "End-to-end pipeline: Tokenize → Fine-tune → ISP",
        "=" * 72,
        f"  pipeline config:    {pipeline_path}",
        f"  run directory:      {resolved['pipeline_run_dir']}",
        f"  input_dir:          {resolved['input_dir']}",
        f"  study / prefix:     {resolved['study_name']}",
        f"  species:            {resolved.get('species', {})}",
        f"  max_input_size:     {resolved.get('max_input_size', 2048)}",
        f"  tokenized dir:      {resolved['tokenized_dir']}",
        f"  finetune dir:       {resolved['finetune_run_dir']}",
        "",
        "  stage configs:",
    ]
    for name, p in stage_config_paths.items():
        lines.append(f"    {name}: {p}")
    lines.append("=" * 72)
    return "\n".join(lines) + "\n"
