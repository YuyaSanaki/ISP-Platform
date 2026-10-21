### English

`run_isp_rank_enrichment.py` (also `--run-dir <run>` by hand): gene set enrichment of the top genes of Start, Goal and ISP.
Pseudobulk rank per gene (`state_feedback/gene_ranks.py`: mean over cells of position / (length − 1), 1 where the gene is
not encoded). Background: genes detected in ≥ `min_detection` (5) cells of Start and Goal (`background: both`) or either,
perturbed genes excluded. Each list is the top `top_fraction` (10%) of the background. ORA (`gene_set_enrichment.py`:
one-sided Fisher, BH) on local Enrichr GMT files (Reactome 2024, GO BP/MF/CC 2025, KEGG 2021 Human or 2019 Mouse).
Normal ISP (per-gene screens): ISP list = genes shifted toward Goal with `Goal_end_FDR` < `fdr`, ranked by shift, capped at
the Start list size, universe = background genes ISP tested; Start and Goal from the ISP dataset (≤ `max_cells_per_state`,
3000). State-feedback ISP (stability runner, one step): ISP list by gene-level Δspec (placebo-corrected rerank minus mean
under comparison placebo chains, averaged over decoder seeds), plus **closeness to Goal** (`state_feedback/closeness.py`:
progress, toward_goal, cosine, Spearman, distance closed, against leave-one-out placebo chains). Figures
(`enrichment_plot.py`): `enrich_<library>` and `closeness` in `rank_enrichment/figures/` (PNG, PDF).

### 日本語

`run_isp_rank_enrichment.py`（`--run-dir <run>` で手動でも実行可）: Start、Goal、ISP の上位遺伝子の遺伝子セット濃縮解析です。
遺伝子ごとの pseudobulk rank（`state_feedback/gene_ranks.py`。細胞ごとの 位置 / (長さ − 1) の平均。エンコードにない遺伝子は 1）を使います。
背景は Start と Goal の両方（`background: both`）またはどちらかで `min_detection`（5）細胞以上に検出された遺伝子から、摂動した遺伝子を除いたものです。
各リストは背景の上位 `top_fraction`（10%）。ORA（`gene_set_enrichment.py`。片側 Fisher、BH）は手元の Enrichr の GMT ファイル
（Reactome 2024、GO BP/MF/CC 2025、KEGG 2021 Human または 2019 Mouse）で計算します。
通常の ISP（遺伝子ごとのスクリーニング）: ISP のリストは Goal 側に動き `Goal_end_FDR` < `fdr` の遺伝子を shift の順に並べ、Start のリストの長さで打ち切ったもの。
母集団は ISP が調べた背景遺伝子。Start と Goal は ISP のデータセットから取ります（状態ごとに最大 `max_cells_per_state`、3000 細胞）。
State-feedback ISP（stability runner、1 段階向け）: ISP のリストは遺伝子ごとの Δspec（プラセボ補正後の並べ替えでの変位 − 比較用プラセボ連鎖での平均変位。デコーダーのシードで平均）の順。
さらに **closeness to Goal**（`state_feedback/closeness.py`。progress、toward_goal、cosine、Spearman、distance closed を leave-one-out のプラセボ連鎖と比較）を出します。
図（`enrichment_plot.py`）は `rank_enrichment/figures/` の `enrich_<library>` と `closeness`（PNG と PDF）。
