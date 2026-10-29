"""Copy-paste Methods paragraph for the Web UI.

No Streamlit and no model stack. Token length is the number of genes kept
per cell after rank tokenization (the dataset ``length`` column).
"""
from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

RUN_KIND_E2E = "e2e"
RUN_KIND_FT_BATCH = "ft_batch"
RUN_KIND_ISP_UMAP = "isp_umap"
RUN_KIND_STATE_FEEDBACK = "state_feedback"

_SPECIES_PROSE = {
    "mouse": "mouse",
    "human": "human",
    "drosophila": "fruit fly (Drosophila)",
}
_NATIVE_ORGANISM = {
    "mouse_geneformer": "mouse",
    "human_geneformer": "human",
}
_MODEL_SPECS: dict[tuple[str, str], tuple[str, int]] = {
    ("mouse_geneformer", "base"): ("Mouse Geneformer base", 2048),
    ("mouse_geneformer", "12l_e20"): ("Mouse Geneformer 12L-E20", 2048),
    ("human_geneformer", "v2_104m"): ("Human Geneformer V2-104M", 4096),
    ("human_geneformer", "v2_316m"): ("Human Geneformer V2-316M", 4096),
}
_ORTHOLOG_BY = {
    "one2one": "one-to-one orthologs",
    "best_of_n": "keeping the highest-median ortholog when several genes mapped to one model gene",
    "legacy_sum": "the legacy rule that sums counts when several genes map to one model gene",
}
_DEFAULT_FT_BATCH = 6
_SHA = re.compile(r"^[0-9a-fA-F]{7,40}$")
_MERGED = re.compile(r"^.+_merged\.dataset$")
_SHARD = re.compile(r"^.+_(\d+)\.dataset$")

_LENGTH_CACHE: dict[tuple, "LengthStats"] = {}


class LengthReadError(RuntimeError):
    """A tokenized dataset is present but its length column could not be read."""


@dataclass(frozen=True)
class LengthStats:
    n: int
    median: float
    mean: float
    sd: float | None
    minimum: int
    maximum: int
    n_at_cap: int | None
    input_size: int | None


@dataclass
class MethodFacts:
    version: str
    run_kind: str
    commit: str | None = None
    source_ready: bool = True
    input_species: str | None = None
    model_id: str | None = None
    variant: str | None = None
    ortholog_policy: str | None = None
    isp_mode: str | None = None
    genes: tuple[str, ...] = ()
    genome_wide: bool = False
    state_key: str | None = None
    start: str | None = None
    goal: str | None = None
    ft_sentence: str | None = None
    length: LengthStats | None = None
    umap_genes: tuple[str, ...] = ()
    sf_mode: str | None = None
    sf_steps: tuple[tuple[str, tuple[str, ...]], ...] = ()
    recommended_batch: int | None = None
    pipeline_config: bool = False


def platform_version(repo_root: Path) -> str:
    """Version string from the checkout ``VERSION`` file, or empty."""
    try:
        text = (repo_root / "VERSION").read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    return text.split()[0] if text else ""


def short_commit(value: object) -> str | None:
    text = str(value or "").strip()
    if not _SHA.fullmatch(text):
        return None
    return text[:7].lower()


def load_run_record(run_dir: Path) -> tuple[dict[str, Any], str | None]:
    """Pipeline config used for a run folder, and a short git commit if recorded."""
    commit = None
    meta_path = run_dir / "pipeline_run_metadata.yaml"
    if meta_path.is_file():
        meta = _load_yaml(meta_path)
        commit = short_commit(meta.get("git_commit"))
    for relative in ("pipeline_config_used.yaml", "stage_configs/isp.yaml"):
        path = run_dir / relative
        if path.is_file():
            return _load_yaml(path), commit
    return {}, commit


def model_input_size_for_config(cfg: Mapping[str, Any] | None) -> int | None:
    """Model window (2048 or 4096) when the config names a known Geneformer."""
    species = (cfg or {}).get("species") if isinstance(cfg, Mapping) else None
    if not isinstance(species, Mapping):
        return None
    model = str(species.get("model") or "").strip().lower()
    if model == "mouse_geneformer":
        variant = str(species.get("mouse_variant") or "base").strip().lower()
    elif model == "human_geneformer":
        variant = str(species.get("human_variant") or "v2_104m").strip().lower()
    else:
        return None
    spec = _MODEL_SPECS.get((model, variant))
    if spec:
        return spec[1]
    return 2048 if model == "mouse_geneformer" else 4096


def tokenized_dataset_dirs(run_dir: Path) -> list[Path]:
    """Dataset directories fine-tune and ISP actually use.

    A merged dataset wins over the shards it was built from.
    """
    root = run_dir / "tokenized_dataset"
    if not root.is_dir():
        return []
    merged = sorted(p for p in root.iterdir() if p.is_dir() and _MERGED.match(p.name))
    if merged:
        return merged
    shards: list[tuple[int, Path]] = []
    for path in root.iterdir():
        match = _SHARD.match(path.name)
        if match and path.is_dir():
            shards.append((int(match.group(1)), path))
    shards.sort()
    return [path for _, path in shards]


def summarize_lengths(
    lengths: Sequence[int],
    *,
    max_input_size: int | None = None,
) -> LengthStats | None:
    """Median, mean and sample SD of token lengths. Empty input returns None."""
    values = [int(v) for v in lengths]
    n = len(values)
    if n == 0:
        return None
    input_size = int(max_input_size) if max_input_size else None
    n_at_cap = None
    if input_size is not None:
        n_at_cap = sum(1 for value in values if value >= input_size)
    return LengthStats(
        n=n,
        median=float(statistics.median(values)),
        mean=float(statistics.fmean(values)),
        sd=float(statistics.stdev(values)) if n >= 2 else None,
        minimum=min(values),
        maximum=max(values),
        n_at_cap=n_at_cap,
        input_size=input_size,
    )


def token_lengths_cached(run_dir: Path, *, max_input_size: int | None = None) -> bool:
    """True when this run's length statistics are already in memory."""
    dirs = tokenized_dataset_dirs(run_dir)
    if not dirs:
        return True
    return _cache_key(dirs, max_input_size, True) in _LENGTH_CACHE


def length_stats_for_run(
    run_dir: Path,
    *,
    max_input_size: int | None = None,
    loader: Callable[[Path], list[int]] | None = None,
) -> LengthStats | None:
    """Length statistics for ``run_dir/tokenized_dataset``, or None if absent.

    Raises ``LengthReadError`` when a dataset directory exists but cannot be read.
    """
    dirs = tokenized_dataset_dirs(run_dir)
    if not dirs:
        return None
    load = loader or _load_length_column
    key = _cache_key(dirs, max_input_size, loader is None)
    if loader is None and key in _LENGTH_CACHE:
        return _LENGTH_CACHE[key]
    lengths: list[int] = []
    for dataset_dir in dirs:
        lengths.extend(int(v) for v in load(dataset_dir))
    stats = summarize_lengths(lengths, max_input_size=max_input_size)
    if stats is None:
        raise LengthReadError(f"{dirs[0].name} has no cells.")
    if loader is None:
        _LENGTH_CACHE[key] = stats
        if len(_LENGTH_CACHE) > 8:
            _LENGTH_CACHE.pop(next(iter(_LENGTH_CACHE)))
    return stats


def facts_from_config(
    cfg: Mapping[str, Any] | None,
    *,
    version: str,
    run_kind: str,
    git_commit: str | None = None,
    source_ready: bool = True,
    length: LengthStats | None = None,
    umap_genes: Sequence[str] | None = None,
    sf_mode: str | None = None,
    sf_steps: Sequence[Mapping[str, Any]] | None = None,
    recommended_batch: int | None = None,
) -> MethodFacts:
    """Collect the fields the Methods paragraph is allowed to state."""
    cfg = cfg if isinstance(cfg, Mapping) else {}
    facts = MethodFacts(
        version=str(version or "").strip(),
        run_kind=run_kind,
        commit=short_commit(git_commit),
        source_ready=source_ready,
        length=length,
        umap_genes=_clean_genes(umap_genes),
        sf_mode=_normalize_sf_mode(sf_mode),
        sf_steps=_clean_steps(sf_steps),
        recommended_batch=_positive_int(recommended_batch),
        pipeline_config=("data" in cfg or "stages" in cfg),
    )
    species = cfg.get("species") if isinstance(cfg.get("species"), Mapping) else None
    if species is not None:
        facts.input_species = str(species.get("model_organism") or "mouse").strip().lower() or None
        facts.model_id = str(species.get("model") or "mouse_geneformer").strip().lower() or None
        if facts.model_id == "human_geneformer":
            facts.variant = str(species.get("human_variant") or "v2_104m").strip().lower()
        elif facts.model_id == "mouse_geneformer":
            facts.variant = str(species.get("mouse_variant") or "base").strip().lower()
        policy = str(species.get("ortholog_policy") or "").strip().lower()
        facts.ortholog_policy = policy or None
    pert = cfg.get("perturbation") if isinstance(cfg.get("perturbation"), Mapping) else None
    if pert is not None and run_kind != RUN_KIND_FT_BATCH:
        mode = str(pert.get("type") or "").strip().lower()
        facts.isp_mode = mode or None
        facts.genes = _clean_genes(pert.get("genes_to_perturb"))
        facts.genome_wide = not facts.genes
        facts.state_key = str(pert.get("state_key") or "").strip() or None
        facts.start = str(pert.get("start_state") or "").strip() or None
        facts.goal = str(pert.get("end_state") or "").strip() or None
    if run_kind != RUN_KIND_FT_BATCH:
        facts.ft_sentence = _ft_sentence(cfg, pipeline_config=facts.pipeline_config)
    return facts


def format_method_description(facts: MethodFacts) -> str:
    """One paragraph, safe to paste into a Methods section."""
    sentences: list[str] = []
    sentences.extend(_opening(facts))
    if facts.run_kind == RUN_KIND_STATE_FEEDBACK:
        sentences.extend(_state_feedback_sentences(facts))
    sentences.extend(_species_sentences(facts))
    model = _model_sentence(facts)
    if model:
        sentences.append(model)
    if facts.length is not None and facts.run_kind != RUN_KIND_FT_BATCH:
        sentences.extend(_length_sentences(facts.length))
    if facts.run_kind == RUN_KIND_ISP_UMAP and facts.umap_genes:
        if len(facts.umap_genes) == 1:
            sentences.append(f"The UMAP perturbed {facts.umap_genes[0]}.")
        else:
            sentences.append(
                f"The UMAP perturbed {_join_genes(facts.umap_genes)} together."
            )
    if facts.run_kind not in (RUN_KIND_FT_BATCH, RUN_KIND_STATE_FEEDBACK) and facts.isp_mode:
        sentences.extend(_isp_sentences(facts))
    if facts.run_kind != RUN_KIND_FT_BATCH and facts.start and facts.goal:
        sentences.append(_state_sentence(facts))
    if facts.run_kind == RUN_KIND_FT_BATCH and facts.recommended_batch:
        sentences.append(
            f"The recommended train_batch_size was {facts.recommended_batch}."
        )
    elif facts.ft_sentence:
        sentences.append(facts.ft_sentence)
    return " ".join(sentences)


def _opening(facts: MethodFacts) -> list[str]:
    name = _version_phrase(facts)
    if facts.run_kind == RUN_KIND_FT_BATCH:
        return [
            f"Fine-tune batch size was calibrated in {name}.",
            "This job did not tokenize cells, fine-tune a model, or run in silico perturbation.",
        ]
    if facts.run_kind == RUN_KIND_ISP_UMAP:
        if facts.source_ready:
            return [
                f"A per-cell trajectory UMAP was drawn in {name} from a completed end-to-end pipeline run."
            ]
        return [f"A per-cell trajectory UMAP is configured in {name}."]
    if facts.run_kind == RUN_KIND_STATE_FEEDBACK:
        if facts.source_ready:
            return [
                f"State-feedback ISP was run in {name} on a completed end-to-end pipeline model."
            ]
        return [f"State-feedback ISP is configured in {name}."]
    return [f"Analyses were run in {name} as an end-to-end pipeline."]


def _version_phrase(facts: MethodFacts) -> str:
    if facts.version and facts.commit:
        return f"ISP Platform v{facts.version} (commit {facts.commit})"
    if facts.version:
        return f"ISP Platform v{facts.version}"
    if facts.commit:
        return f"ISP Platform (commit {facts.commit})"
    return "ISP Platform"


def _species_sentences(facts: MethodFacts) -> list[str]:
    if not facts.input_species:
        return []
    label = _SPECIES_PROSE.get(facts.input_species, facts.input_species)
    sentences = [f"The input species was {label}."]
    native = _NATIVE_ORGANISM.get(facts.model_id or "")
    if native and facts.input_species != native:
        how = _ORTHOLOG_BY.get(facts.ortholog_policy or "")
        if how:
            sentences.append(f"Genes were mapped to the {native} model vocabulary by {how}.")
        else:
            sentences.append(f"Genes were mapped to the {native} model vocabulary.")
    return sentences


def _model_sentence(facts: MethodFacts) -> str | None:
    if not facts.model_id:
        return None
    spec = _MODEL_SPECS.get((facts.model_id, facts.variant or ""))
    if spec:
        return f"The model was {spec[0]}."
    if facts.model_id == "mouse_geneformer":
        return "The model was Mouse Geneformer."
    if facts.model_id == "human_geneformer":
        return "The model was Human Geneformer."
    if facts.variant:
        return f"The model was {facts.model_id} ({facts.variant})."
    return f"The model was {facts.model_id}."


def _length_sentences(stats: LengthStats) -> list[str]:
    details = [f"mean {_fmt_decimal(stats.mean)}"]
    if stats.sd is not None:
        details.append(f"SD {_fmt_decimal(stats.sd)}")
    details.append(f"range {_fmt_int(stats.minimum)}–{_fmt_int(stats.maximum)}")
    details.append(f"n = {_fmt_int(stats.n)} cells")
    window = ""
    if stats.input_size:
        window = f", truncated at {_fmt_int(stats.input_size)}"
    sentences = [
        "Token length (genes kept per cell after ranking"
        f"{window}) had median {_fmt_median(stats.median)} ({', '.join(details)})."
    ]
    if stats.n_at_cap and stats.input_size:
        sentences.append(
            f"{_fmt_int(stats.n_at_cap)} cells were at the input-size cap of "
            f"{_fmt_int(stats.input_size)}."
        )
    return sentences


def _isp_sentences(facts: MethodFacts) -> list[str]:
    sentences: list[str] = []
    if facts.isp_mode:
        # Genome-wide is part of the mode. A named UMAP gene is one perturbation, not the screen.
        wide = facts.genome_wide and facts.run_kind != RUN_KIND_ISP_UMAP
        label = f"Genome-wide {facts.isp_mode}" if wide else facts.isp_mode
        sentences.append(f"ISP mode was {label}.")
    if facts.run_kind == RUN_KIND_ISP_UMAP:
        return sentences
    if facts.genes and not facts.genome_wide:
        sentences.append(f"The genes perturbed together were {_join_genes(facts.genes)}.")
    return sentences


def _state_sentence(facts: MethodFacts) -> str:
    if facts.state_key:
        return (
            f"Start was {facts.start} and Goal was {facts.goal} ({facts.state_key})."
        )
    return f"Start was {facts.start} and Goal was {facts.goal}."


def _state_feedback_sentences(facts: MethodFacts) -> list[str]:
    sentences: list[str] = []
    if facts.sf_mode == "placebo":
        sentences.append("The run used placebo correction.")
    elif facts.sf_mode == "diagnostic":
        sentences.append("The run used decoder diagnostics without placebo correction.")
    for index, (ptype, genes) in enumerate(facts.sf_steps, start=1):
        if not genes:
            continue
        sentences.append(f"Step {index} was {ptype} of {_join_genes(genes)}.")
    return sentences


def _auto_measurement_clause(runtime: Mapping[str, Any]) -> str:
    """How Auto chose the batch, from uploaded cells and free memory."""
    if str(runtime.get("ft_batch_choice") or "").strip().lower() != "auto":
        return ""
    n_cells = _positive_int(runtime.get("ft_n_cells"))
    length_max = _positive_int(runtime.get("ft_length_max"))
    model_length = _positive_int(runtime.get("ft_input_length"))
    try:
        memory_gib = float(runtime.get("ft_memory_gib") or 0)
    except (TypeError, ValueError):
        memory_gib = 0.0
    parts: list[str] = []
    missing = ""
    if n_cells and length_max:
        parts.append(f"{n_cells} uploaded cells with longest rank length {length_max}")
    elif model_length:
        parts.append(f"the model input length {model_length}")
        missing = " No count matrix was loaded."
    if memory_gib:
        parts.append(f"{memory_gib:.0f} GiB of available memory")
    if not parts:
        return " This batch was selected automatically from the input length."
    return (
        " This batch was selected automatically from "
        + " and ".join(parts)
        + "."
        + missing
    )


def _ft_sentence(cfg: Mapping[str, Any], *, pipeline_config: bool) -> str | None:
    runtime = cfg.get("runtime") if isinstance(cfg.get("runtime"), Mapping) else {}
    training = (
        ((cfg.get("stages") or {}).get("finetune") or {}).get("training")
        if isinstance(cfg.get("stages"), Mapping)
        else {}
    )
    if not isinstance(training, Mapping):
        training = {}
    choice = str(runtime.get("ft_batch_choice") or "").strip().lower()
    mode = str(runtime.get("ft_batch_mode") or "").strip().lower()
    micro = _positive_int(runtime.get("micro_batch_size"))
    effective = _positive_int(runtime.get("micro_effective_batch_size"))
    accum = _positive_int(runtime.get("micro_gradient_accumulation_steps"))
    seq_len = _positive_int(runtime.get("ft_input_length"))
    decided = bool(runtime.get("ft_batch_decided"))
    if choice == "auto" and not decided:
        size = effective or _DEFAULT_FT_BATCH
        return (
            f"Fine-tuning keeps an effective batch size of {size}. "
            "When fine-tuning starts, the run measures the token lengths of the cells in that job "
            "and the free memory on the machine that runs it. "
            "One GPU pass of that many cells is used when it fits. "
            "When it does not fit, each weight update is split into GPU passes that divide that size."
        )
    if mode == "micro" and micro and (effective or accum):
        passes = accum or (effective // micro if effective and effective % micro == 0 else None)
        update_cells = effective or (micro * passes if passes else None)
        if passes and update_cells:
            sentence = (
                f"Fine-tuning used an effective batch size of {update_cells}. "
                f"Each weight update accumulated {passes} GPU passes of {micro} cells."
            )
            if choice == "auto" and seq_len:
                sentence += (
                    f" A single pass of {update_cells} cells did not fit in GPU memory "
                    f"at rank length {seq_len}."
                )
            return sentence + _auto_measurement_clause(runtime)
    explicit = _positive_int(runtime.get("train_batch_size"))
    if explicit is None:
        explicit = _positive_int(training.get("batch_size"))
    grad_accum = _positive_int(runtime.get("gradient_accumulation_steps"))
    if mode not in ("standard", "micro") and grad_accum and grad_accum > 1 and explicit:
        return (
            f"Fine-tuning used batches of {explicit} cells, accumulated over {grad_accum} "
            f"passes ({explicit * grad_accum} cells per weight update)."
        )
    if choice == "auto" and (effective or explicit):
        size = effective or explicit
        sentence = (
            f"Fine-tuning used a batch size of {size}. "
            f"Each weight update used one GPU pass of {size} cells."
        )
        if seq_len:
            sentence += f" A pass of {size} cells fit at rank length {seq_len}."
        return sentence + _auto_measurement_clause(runtime)
    if explicit:
        return (
            f"Fine-tuning used a standard batch of {explicit} cells per weight update "
            f"(train_batch_size {explicit})."
        )
    if pipeline_config or mode == "standard":
        return f"Fine-tuning used a batch size of {_DEFAULT_FT_BATCH}."
    return None


def _normalize_sf_mode(mode: str | None) -> str | None:
    text = str(mode or "").strip().lower()
    if "diagnostic" in text:
        return "diagnostic"
    if "placebo" in text:
        return "placebo"
    return None


def _clean_steps(
    steps: Sequence[Mapping[str, Any]] | None,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    cleaned: list[tuple[str, tuple[str, ...]]] = []
    for step in steps or []:
        if not isinstance(step, Mapping):
            continue
        ptype = str(step.get("type") or "").strip().lower() or "overexpress"
        cleaned.append((ptype, _clean_genes(step.get("genes"))))
    return tuple(cleaned)


def _clean_genes(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw = [part.strip() for part in value.replace(",", "\n").splitlines()]
    else:
        try:
            raw = [str(item).strip() for item in value]  # type: ignore[union-attr]
        except TypeError:
            raw = [str(value).strip()]
    seen: list[str] = []
    for gene in raw:
        if gene and gene not in seen:
            seen.append(gene)
    return tuple(seen)


def _join_genes(genes: Sequence[str]) -> str:
    names = [gene for gene in genes if gene]
    if len(names) > 12:
        return f"{', '.join(names[:8])} and {len(names) - 8} further genes"
    return _join_and(names)


def _join_and(names: Sequence[str]) -> str:
    items = list(names)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _fmt_int(value: int) -> str:
    return f"{int(value):,}"


def _fmt_decimal(value: float) -> str:
    return f"{value:,.1f}"


def _fmt_median(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return _fmt_int(int(round(value)))
    return f"{value:,.1f}"


def _positive_int(value: object) -> int | None:
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number >= 1 else None


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _cache_key(
    dirs: Sequence[Path],
    max_input_size: int | None,
    use_cache: bool,
) -> tuple:
    if not use_cache:
        return ()
    parts: list[tuple] = []
    for dataset_dir in dirs:
        try:
            parts.append(("dir", str(dataset_dir), dataset_dir.stat().st_mtime_ns))
        except OSError:
            parts.append(("dir", str(dataset_dir), 0))
        for arrow in _arrow_files(dataset_dir):
            try:
                stat = arrow.stat()
            except OSError:
                continue
            parts.append((str(arrow), stat.st_mtime_ns, stat.st_size))
    return (max_input_size, tuple(parts))


def _arrow_files(dataset_dir: Path) -> list[Path]:
    files: list[Path] = []
    state_path = dataset_dir / "state.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {}
        for item in state.get("_data_files") or []:
            name = item.get("filename") if isinstance(item, dict) else None
            if name:
                files.append(dataset_dir / str(name))
    if not files:
        files = sorted(dataset_dir.glob("*.arrow"))
    return files


def _load_length_column(dataset_dir: Path) -> list[int]:
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise LengthReadError(
            "Token lengths were not read because the datasets package is not available."
        ) from exc
    try:
        dataset = load_from_disk(str(dataset_dir))
    except Exception as exc:
        raise LengthReadError(
            f"Token lengths could not be read from {dataset_dir.name}."
        ) from exc
    if not hasattr(dataset, "column_names"):
        keys = list(dataset.keys())
        if not keys:
            raise LengthReadError(f"{dataset_dir.name} has no rows.")
        dataset = dataset["train"] if "train" in keys else dataset[keys[0]]
    if "length" not in list(dataset.column_names or []):
        raise LengthReadError(f"{dataset_dir.name} has no length column.")
    return [int(value) for value in dataset["length"]]


def apply_form_overrides(
    cfg: Mapping[str, Any] | None,
    form: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Copy a pipeline config and cover it with Web UI widget values."""
    out = dict(cfg or {})
    form = form or {}
    species = dict(out.get("species") or {})
    for form_key, field_name in (
        ("model_organism", "model_organism"),
        ("model", "model"),
        ("mouse_variant", "mouse_variant"),
        ("human_variant", "human_variant"),
        ("ortholog_policy", "ortholog_policy"),
    ):
        if form.get(form_key):
            species[field_name] = form[form_key]
    if species:
        out["species"] = species

    pert = dict(out.get("perturbation") or {})
    if "pert_type" in form:
        pert["type"] = form.get("pert_type")
    if "genes" in form:
        pert["genes_to_perturb"] = list(form.get("genes") or [])
    if form.get("start"):
        pert["start_state"] = form["start"]
    if form.get("goal"):
        pert["end_state"] = form["goal"]
    if form.get("state_key"):
        pert["state_key"] = form["state_key"]
    if pert:
        out["perturbation"] = pert

    runtime = dict(out.get("runtime") or {})
    if form.get("ft_batch_choice"):
        runtime["ft_batch_choice"] = form["ft_batch_choice"]
    if form.get("ft_batch_mode"):
        runtime["ft_batch_mode"] = form["ft_batch_mode"]
    if form.get("train_batch_size"):
        runtime["train_batch_size"] = form["train_batch_size"]
    if form.get("micro_batch_size"):
        runtime["micro_batch_size"] = form["micro_batch_size"]
    if form.get("micro_effective_batch_size"):
        runtime["micro_effective_batch_size"] = form["micro_effective_batch_size"]
    if form.get("micro_accum_steps"):
        runtime["micro_gradient_accumulation_steps"] = form["micro_accum_steps"]
    if form.get("ft_input_length"):
        runtime["ft_input_length"] = form["ft_input_length"]
    if form.get("ft_n_cells"):
        runtime["ft_n_cells"] = form["ft_n_cells"]
    if form.get("ft_length_max"):
        runtime["ft_length_max"] = form["ft_length_max"]
    if form.get("ft_memory_gib"):
        runtime["ft_memory_gib"] = form["ft_memory_gib"]
    if runtime:
        out["runtime"] = runtime
    return out


def run_record_ready(run_dir: Path | None) -> bool:
    if run_dir is None or not run_dir.is_dir():
        return False
    return (run_dir / "pipeline_config_used.yaml").is_file() or (
        run_dir / "stage_configs" / "isp.yaml"
    ).is_file()


def build_method_view(
    *,
    version: str,
    run_kind: str,
    recorded_only: bool = False,
    run_dir: Path | None = None,
    pipeline_run_dir: Path | None = None,
    live_cfg: Mapping[str, Any] | None = None,
    busy: bool = False,
    umap_source: Path | None = None,
    umap_genes: Sequence[str] | None = None,
    sf_source: Path | None = None,
    sf_mode: str | None = None,
    sf_steps: Sequence[Mapping[str, Any]] | None = None,
    recommended_batch: int | None = None,
    length_reader: Callable[[Path, Mapping[str, Any]], LengthStats | None] | None = None,
) -> tuple[str, list[str]]:
    """Paragraph plus captions that stay outside the copied text."""
    notes: list[str] = []
    kind = RUN_KIND_E2E if recorded_only else run_kind
    source_ready = True
    used_disk = False
    commit: str | None = None
    cfg: Mapping[str, Any] = {}
    active: Path | None = None

    def take_disk(path: Path | None) -> bool:
        nonlocal cfg, commit, used_disk, active
        if not run_record_ready(path):
            return False
        assert path is not None
        cfg, commit = load_run_record(path)
        used_disk = bool(cfg)
        active = path
        return used_disk

    if recorded_only:
        if not take_disk(run_dir) and run_dir is not None:
            notes.append("This run has no saved pipeline config, so model settings are omitted.")
            active = run_dir
    elif kind == RUN_KIND_E2E:
        disk = run_dir or pipeline_run_dir
        if not take_disk(disk):
            cfg = dict(live_cfg or {})
            active = disk if disk is not None and disk.is_dir() else None
    elif kind == RUN_KIND_FT_BATCH:
        cfg = dict(live_cfg or {})
    else:
        source = umap_source if kind == RUN_KIND_ISP_UMAP else sf_source
        if not take_disk(source):
            source_ready = False
            active = source if source is not None and source.is_dir() else None

    length = None
    length_error = None
    if active is not None and kind != RUN_KIND_FT_BATCH and (active / "tokenized_dataset").is_dir():
        reader = length_reader or _default_length_reader
        try:
            length = reader(active, cfg)
        except LengthReadError as exc:
            length_error = str(exc)
        except OSError:
            length_error = "Token lengths could not be read from this run folder."

    text = format_method_description(
        facts_from_config(
            cfg,
            version=version,
            run_kind=kind,
            git_commit=commit,
            source_ready=source_ready,
            length=length,
            umap_genes=umap_genes if kind == RUN_KIND_ISP_UMAP else None,
            sf_mode=sf_mode if kind == RUN_KIND_STATE_FEEDBACK else None,
            sf_steps=sf_steps if kind == RUN_KIND_STATE_FEEDBACK else None,
            recommended_batch=recommended_batch if kind == RUN_KIND_FT_BATCH else None,
        )
    )

    if kind == RUN_KIND_FT_BATCH and not _positive_int(recommended_batch):
        notes.append("The recommended batch size is added after this calibration finishes.")
    if kind in (RUN_KIND_ISP_UMAP, RUN_KIND_STATE_FEEDBACK) and not source_ready:
        notes.append(
            "Select a past pipeline run to fill species, model, token lengths, "
            "Start, Goal and the fine-tune batch."
        )
    if kind == RUN_KIND_ISP_UMAP and source_ready and not _clean_genes(umap_genes):
        notes.append("Select a gene to name it in the paragraph.")
    if kind == RUN_KIND_STATE_FEEDBACK and sf_steps and any(
        not _clean_genes(step.get("genes") if isinstance(step, Mapping) else None)
        for step in sf_steps
    ):
        notes.append("Steps with no genes are left out of the paragraph.")
    if used_disk and active is not None:
        notes.append(f"This paragraph follows `{active.name}`.")
    elif kind == RUN_KIND_E2E:
        notes.append("This paragraph follows the current form.")
    if length is None and kind != RUN_KIND_FT_BATCH and (kind == RUN_KIND_E2E or source_ready):
        if length_error and not busy:
            notes.append(length_error)
        else:
            notes.append(
                "Token lengths are added after tokenization writes tokenized_dataset/."
            )
    return text, notes


def render_method_box(
    st: Any,
    text: str,
    notes: Sequence[str],
    *,
    key_prefix: str,
    height: int = 280,
) -> None:
    """Text box the author can select and copy. Edits are not written anywhere."""
    import hashlib

    st.markdown("**Method description**")
    st.caption(
        "Copy into Methods. Token length is the number of genes kept per cell after ranking. "
        "Edits in the box are not saved."
    )
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    st.text_area(
        "Method description",
        value=text,
        height=height,
        key=f"{key_prefix}_method_{digest}",
        label_visibility="collapsed",
        help="Select this paragraph and copy it into the paper. The box is not saved.",
    )
    for note in notes:
        st.caption(note)


def _default_length_reader(run_dir: Path, cfg: Mapping[str, Any]) -> LengthStats | None:
    return length_stats_for_run(run_dir, max_input_size=model_input_size_for_config(cfg))
