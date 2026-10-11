### English

Ortholog conversion between mouse↔human and drosophila→human/mouse using `dicts/orthologs/{pair}.tsv`.
Policies: `one2one` (default; keep only unique one-to-one pairs, never sum), `best_of_n` (pick one target; for many sources
keep the highest-median source), `legacy_sum` (last row wins, expression summed). Merge order: main table → platform
`*_curated.tsv` → optional project overlay (`GENEFORMER_ORTHOLOG_CURATED_OVERLAY`). `_resolve_gene_id` is shared with ISP
so perturbation gene names resolve the same way as at tokenize.

### 日本語

`dicts/orthologs/{pair}.tsv` を使い、マウス↔ヒト、ショウジョウバエ→ヒト/マウスのオルソログ変換を行います。
方針は `one2one`（既定。1対1の組だけ残し、合算しない）、`best_of_n`（変換先を1つ選ぶ。変換元が複数なら中央値発現が最大のものを残す）、
`legacy_sum`（最後の行が勝ち、発現を合算）の3つ。結合順は、本体の表 → プラットフォームの `*_curated.tsv` → 任意のプロジェクト用オーバーレイ
（`GENEFORMER_ORTHOLOG_CURATED_OVERLAY`）。`_resolve_gene_id` は ISP と共用で、摂動する遺伝子名もトークン化と同じ方法で解決されます。
