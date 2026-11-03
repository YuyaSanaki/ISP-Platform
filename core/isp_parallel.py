"""Opt-in parallel genome-wide ISP.

Default ISP stays one process. Parallel mode is selected only when a chunk id
and a chunk count are both set. Each process perturbs a stride of the
start-state cells, writes its own pickle files, and skips statistics.
``run_isp_merge.py`` combines the pickles and then runs statistics once.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class ChunkMergeError(ValueError):
    """The chunk pickles cannot be combined into one statistics table."""


def chunk_tag(chunk_id: int, chunk_count: int) -> str:
    return f"chunk{chunk_id}of{chunk_count}"


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _parse_layer(chunk_id: Any, chunk_count: Any, layer: str) -> tuple[int, int] | None:
    if _blank(chunk_id) and _blank(chunk_count):
        return None
    if _blank(chunk_id) or _blank(chunk_count):
        raise ValueError(
            f"{layer}: set both the chunk id and the chunk count, or leave both unset "
            "(unset is the default single-process ISP)."
        )
    try:
        index = int(chunk_id)
        count = int(chunk_count)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{layer}: chunk id and chunk count must be integers.") from exc
    if count < 2:
        raise ValueError(
            f"{layer}: chunk count must be at least 2. "
            "Leave the chunk id and count unset for single-process ISP."
        )
    if not 0 <= index < count:
        raise ValueError(f"{layer}: chunk id must be from 0 through {count - 1}, got {index}.")
    return index, count


def parse_chunk_spec(
    cli_id: Any = None,
    cli_count: Any = None,
    env_id: Any = None,
    env_count: Any = None,
    yaml_id: Any = None,
    yaml_count: Any = None,
) -> tuple[int, int] | None:
    """CLI, then environment, then YAML. None means single-process ISP."""
    for layer, pair in (
        ("--chunk-id/--chunk-count", (cli_id, cli_count)),
        ("ISP_CHUNK_ID/ISP_CHUNK_COUNT", (env_id, env_count)),
        ("isp.chunk in YAML", (yaml_id, yaml_count)),
    ):
        parsed = _parse_layer(pair[0], pair[1], layer)
        if parsed is not None:
            return parsed
    return None


def chunk_shares_output_folder(explicit_subdir: str | None, auto_time_subdir: bool) -> bool:
    """True when every chunk writes one folder.

    A fixed subdir name does that. Turning off the automatic time folder does too,
    which is how Pipeline (E2E) keeps every chunk in the pipeline run directory.
    """
    return bool(explicit_subdir) or not auto_time_subdir


def yaml_chunk_pair(cfg: Mapping[str, Any] | None) -> tuple[Any, Any]:
    isp = (cfg or {}).get("isp") or {}
    chunk = isp.get("chunk") or {}
    if not isinstance(chunk, dict):
        raise ValueError("isp.chunk must be a mapping with id and count, or omitted.")
    return chunk.get("id"), chunk.get("count")


def validate_cell_shard(cell_shard: Any) -> tuple[int, int] | None:
    if cell_shard is None:
        return None
    if not isinstance(cell_shard, (tuple, list)) or len(cell_shard) != 2:
        raise ValueError("cell_shard must be (index, count) or omitted.")
    return _parse_layer(cell_shard[0], cell_shard[1], "cell_shard")


def assert_parallel_shard_allowed(
    *,
    perturb_group: bool,
    combos: int,
    anchor_token: Any,
    cell_inds_to_perturb: Any,
    num_processes: int,
) -> None:
    if perturb_group or int(combos or 0) != 0 or anchor_token is not None:
        raise ValueError(
            "Parallel mode only splits a genome-wide per-gene screen "
            "(empty genes_to_perturb, combos 0, no anchor gene)."
        )
    if cell_inds_to_perturb != "all":
        raise ValueError("cell_shard cannot be combined with cell_inds_to_perturb.")
    if int(num_processes) > 1:
        raise ValueError(
            "cell_shard cannot be combined with a multi-process accelerate launch. "
            "Start one process per chunk instead."
        )


def reject_parallel_mode(*, genes_to_perturb: list, combos: int, anchor_gene: Any) -> None:
    """Config-level check before the perturber is built."""
    if list(genes_to_perturb or []):
        raise ValueError(
            "Parallel mode is genome-wide only. genes_to_perturb must be empty "
            "(a gene list is perturbed together, which is group mode)."
        )
    if int(combos or 0) != 0:
        raise ValueError("Parallel mode does not support isp.combos.")
    if anchor_gene:
        raise ValueError("Parallel mode does not support isp.anchor_gene.")


def stride_cell_indices(n_cells: int, shard_index: int, shard_count: int) -> list[int]:
    """Cells i where i % shard_count == shard_index, after length-descending sort.

    A contiguous slice would give chunk 0 the longest cells. Stride keeps the
    lengths mixed. Every chunk is non-empty when n_cells >= shard_count.
    """
    if n_cells < shard_count:
        raise ValueError(
            f"Parallel ISP needs at least {shard_count} start cells after downsampling, "
            f"found {n_cells}. Lower the chunk count or raise isp.max_ncells."
        )
    return [i for i in range(n_cells) if i % shard_count == shard_index]


def chunk_config_sha256(cfg: Mapping[str, Any], forward_batch_size: Any) -> str:
    """Hash the settings that change perturbation outputs. Chunk id is not included."""
    paths = cfg.get("paths") or {}
    pert = cfg.get("perturbation") or {}
    model = cfg.get("model") or {}
    isp = dict(cfg.get("isp") or {})
    isp.pop("chunk", None)
    stats = cfg.get("stats") or {}
    payload = {
        "dataset": paths.get("dataset"),
        "geneformer_model": paths.get("geneformer_model"),
        "species": cfg.get("species") or {},
        "perturbation": {
            "type": pert.get("type", "delete"),
            "organ_data": pert.get("organ_data", "experiment"),
            "state_key": pert.get("state_key", "state"),
            "start_state": pert.get("start_state"),
            "end_state": pert.get("end_state"),
            "alt_states": list(pert.get("alt_states") or []),
            "genes_to_perturb": list(pert.get("genes_to_perturb") or []),
        },
        "model": {
            "type": model.get("type", "Pretrained"),
            "num_classes": model.get("num_classes", 0),
            "max_input_size": model.get("max_input_size"),
        },
        "isp": isp,
        "stats_mode": stats.get("mode", "goal_state_shift"),
        "forward_batch_size": None if forward_batch_size is None else str(forward_batch_size),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _as_float_vector(value: Any):
    import numpy as np

    if hasattr(value, "detach"):
        array = value.detach().float().cpu().numpy()
    else:
        array = np.asarray(value, dtype=np.float64)
    return np.asarray(array, dtype=np.float64).reshape(-1)


def centroid_signature(state_embs: Mapping[str, Any] | None) -> tuple[str, dict[str, float]]:
    """SHA-256 of centroids rounded to 3 decimals, plus L2 norms for the error text.

    Rounding absorbs GPU noise around 1e-4. A different state still changes the hash.
    """
    if not state_embs:
        digest = hashlib.sha256(b"none").hexdigest()
        return digest, {}
    import numpy as np

    hasher = hashlib.sha256()
    norms: dict[str, float] = {}
    for key in sorted(state_embs, key=str):
        vector = _as_float_vector(state_embs[key])
        rounded = np.round(vector, 3)
        hasher.update(str(key).encode("utf-8"))
        hasher.update(np.ascontiguousarray(rounded).tobytes())
        hasher.update(str(rounded.shape).encode("utf-8"))
        norms[str(key)] = round(float(np.linalg.norm(vector)), 5)
    return hasher.hexdigest(), norms


def isp_output_prefix(perturb_type: str, organ_data: str, start_state: str, end_state: str) -> str:
    return "output_in-silico_SE{}_OR{}_ST{}_EN{}".format(
        perturb_type,
        organ_data,
        str(start_state).replace(" ", "-"),
        str(end_state).replace(" ", "-"),
    )


def chunk_pickle_glob(perturb_type: str, output_prefix: str, chunk_id: int, chunk_count: int) -> str:
    tag = chunk_tag(chunk_id, chunk_count)
    return f"in_silico_{perturb_type}_{output_prefix}_{tag}_dict_1Kbatch*_raw.pickle"


def clear_chunk_outputs(
    isp_dir: Path,
    marker_dir: Path,
    *,
    perturb_type: str,
    output_prefix: str,
    chunk_id: int,
    chunk_count: int,
) -> list[str]:
    """Delete this chunk's previous pickles and done marker so a rerun can replace them."""
    removed: list[str] = []
    if isp_dir.is_dir():
        pattern = chunk_pickle_glob(perturb_type, output_prefix, chunk_id, chunk_count)
        for path in sorted(isp_dir.glob(pattern)):
            path.unlink()
            removed.append(path.name)
    marker = marker_dir / f"chunk_{chunk_id}_of_{chunk_count}.json"
    if marker.is_file():
        marker.unlink()
        removed.append(marker.name)
    return removed


def write_chunk_marker(marker_dir: Path, marker: dict[str, Any]) -> Path:
    marker_dir.mkdir(parents=True, exist_ok=True)
    index = int(marker["chunk_id"])
    count = int(marker["chunk_count"])
    path = marker_dir / f"chunk_{index}_of_{count}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _load_marker(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        marker = json.load(handle)
    if not isinstance(marker, dict):
        raise ChunkMergeError(f"{path.name} is not a JSON object.")
    if int(marker.get("version", 0)) != 1:
        raise ChunkMergeError(f"{path.name} has unsupported version {marker.get('version')!r}.")
    return marker


def validate_parallel_run(isp_dir: Path, marker_dir: Path, chunk_count: int) -> list[dict[str, Any]]:
    """Require every chunk marker, matching hashes, and exactly the listed pickle files."""
    if chunk_count < 2:
        raise ChunkMergeError("chunk count must be at least 2.")
    missing = [
        i for i in range(chunk_count)
        if not (marker_dir / f"chunk_{i}_of_{chunk_count}.json").is_file()
    ]
    if missing:
        raise ChunkMergeError(
            f"Missing chunk markers for id {missing} under {marker_dir}. "
            "Wait until every chunk process has finished, then rerun any that failed."
        )
    markers = [
        _load_marker(marker_dir / f"chunk_{i}_of_{chunk_count}.json")
        for i in range(chunk_count)
    ]
    for index, marker in enumerate(markers):
        if int(marker.get("chunk_id")) != index or int(marker.get("chunk_count")) != chunk_count:
            raise ChunkMergeError(
                f"chunk_{index}_of_{chunk_count}.json says chunk "
                f"{marker.get('chunk_id')} of {marker.get('chunk_count')}."
            )
    config_hashes = {marker.get("config_sha256") for marker in markers}
    if len(config_hashes) != 1 or None in config_hashes:
        raise ChunkMergeError(
            "Chunk config hashes differ. Rerun every chunk from the same YAML and forward batch size."
        )
    centroid_hashes = {marker.get("centroid_sha256") for marker in markers}
    if len(centroid_hashes) != 1 or None in centroid_hashes:
        norms = [marker.get("centroid_l2") for marker in markers]
        raise ChunkMergeError(
            "Chunk centroid hashes differ, so the workers did not score against the same "
            f"state centroids. L2 norms by chunk: {norms}. "
            "Set one explicit runtime.forward_batch_size and rerun."
        )
    totals = {int(marker.get("n_cells_before_shard")) for marker in markers}
    if len(totals) != 1:
        raise ChunkMergeError(f"Chunks disagree on the start-cell total: {sorted(totals)}.")
    n_before = totals.pop()
    n_sum = sum(int(marker.get("n_cells") or 0) for marker in markers)
    if n_sum != n_before:
        raise ChunkMergeError(
            f"Chunk cell counts sum to {n_sum}, expected {n_before} start cells."
        )
    prefixes = {marker.get("output_prefix") for marker in markers}
    if len(prefixes) != 1:
        raise ChunkMergeError(f"Chunks used different output prefixes: {sorted(prefixes)}.")

    expected: set[str] = set()
    for marker in markers:
        names = list(marker.get("pickle_files") or [])
        if not names:
            raise ChunkMergeError(
                f"Chunk {marker.get('chunk_id')} recorded no pickle files."
            )
        tag = f"_{chunk_tag(int(marker['chunk_id']), chunk_count)}_"
        for name in names:
            if not str(name).endswith("_raw.pickle") or tag not in str(name):
                raise ChunkMergeError(
                    f"Chunk {marker.get('chunk_id')} listed a file outside its chunk: {name}."
                )
            if name in expected:
                raise ChunkMergeError(f"Pickle file listed by more than one chunk: {name}.")
            expected.add(str(name))
            if not (isp_dir / name).is_file():
                raise ChunkMergeError(f"Listed pickle is missing: {isp_dir / name}.")
    if not isp_dir.is_dir():
        raise ChunkMergeError(f"ISP results directory not found: {isp_dir}.")
    actual = {path.name for path in isp_dir.glob("*_raw.pickle")}
    extra = sorted(actual - expected)
    absent = sorted(expected - actual)
    if extra or absent:
        raise ChunkMergeError(
            "Pickle files do not match the chunk markers. "
            f"Extra: {extra or 'none'}. Missing: {absent or 'none'}. "
            "Use a fresh --output-subdir for a parallel run."
        )
    return markers
