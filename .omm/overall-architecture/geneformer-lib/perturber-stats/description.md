### English

`InSilicoPerturberStats` (default mode `goal_state_shift`) reads every `*_raw.pickle`. Genome-wide: per-gene mean shift,
Wilcoxon rank-sum vs pooled shifts of all genes (downsampled to 100k), Benjamini–Hochberg FDR, `N_Detections`, `Sig` = FDR<0.05.
Group mode: one row per cell, no p-values.

### 日本語

`InSilicoPerturberStats`（既定モードは `goal_state_shift`）は `*_raw.pickle` をすべて読みます。ゲノムワイドでは遺伝子ごとの平均 shift、
全遺伝子をまとめた shift（10万件に間引き）との Wilcoxon 順位和検定、Benjamini–Hochberg の FDR、`N_Detections`、`Sig`（FDR<0.05）を出します。
グループモードは細胞ごとに1行で、p 値はありません。
