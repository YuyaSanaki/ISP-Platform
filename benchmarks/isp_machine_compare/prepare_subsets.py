"""Freeze ISP benchmark subsets from one tokenized dataset.

Cells keep their real token lengths. The long subset is cells at the model cap.
The representative subset spreads start cells across the length distribution and
keeps one short and one long goal cell so state centroids still exist.
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import load_from_disk


def _pick(rows: list[dict], n: int, mode: str) -> list[dict]:
    rows = sorted(rows, key=lambda row: (row["length"], row["source_index"]))
    if not rows or n <= 0:
        return []
    if mode == "longest":
        return rows[-n:]
    if n >= len(rows):
        return rows
    # Spread across the sorted lengths, including both ends.
    indexes = [round(i * (len(rows) - 1) / (n - 1)) for i in range(n)]
    chosen = []
    seen = set()
    for index in indexes:
        if index not in seen:
            chosen.append(rows[index])
            seen.add(index)
    return chosen


def main() -> None:
    source = Path(
        "/app/output/20261029/pipeline_070426_493241Z/tokenized_dataset/20261029Matrigel_0.dataset"
    )
    out_root = Path("/app/output/benchmarks/isp_machine_compare/subsets")
    out_root.mkdir(parents=True, exist_ok=True)
    data = load_from_disk(str(source))
    columns = ["input_ids", "length", "disease", "sample_id"]
    start_rows = []
    goal_rows = []
    for index in range(len(data)):
        row = {name: data[index][name] for name in columns}
        row["source_index"] = index
        if row["disease"] == "Matrigel":
            start_rows.append(row)
        elif row["disease"] == "No_Matrigel":
            goal_rows.append(row)

    specs = {
        "long1": {
            "start": _pick([row for row in start_rows if row["length"] >= 4096], 1, "longest"),
            "goal": _pick([row for row in goal_rows if row["length"] >= 4096], 1, "longest"),
        },
        "repr": {
            "start": _pick(start_rows, 4, "spread"),
            "goal": _pick(goal_rows, 2, "spread"),
        },
    }
    manifest = {"source": str(source), "subsets": {}}
    for name, spec in specs.items():
        chosen = spec["start"] + spec["goal"]
        table = {column: [row[column] for row in chosen] for column in columns}
        table["source_index"] = [row["source_index"] for row in chosen]
        subset = data.select([row["source_index"] for row in chosen])
        # Keep only the columns the ISP path reads, plus the source row.
        keep = [column for column in subset.column_names if column in columns or column == "length"]
        # select() preserves all columns. Drop nothing required; save full rows.
        dest = out_root / name
        if dest.exists():
            import shutil
            shutil.rmtree(dest)
        subset.save_to_disk(str(dest))
        start_lengths = [row["length"] for row in spec["start"]]
        manifest["subsets"][name] = {
            "n_start": len(spec["start"]),
            "n_goal": len(spec["goal"]),
            "start_lengths": start_lengths,
            "goal_lengths": [row["length"] for row in spec["goal"]],
            "perturbation_pairs": int(sum(start_lengths)),
            "source_indexes": [row["source_index"] for row in chosen],
            "sample_ids": [row["sample_id"] for row in chosen],
        }
        print(name, manifest["subsets"][name])
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("wrote", out_root / "manifest.json")


if __name__ == "__main__":
    main()
