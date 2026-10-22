### English

Loads the fine-tuned model (fp32, or bf16 with `umap.readout: isp`), resolves gene tokens, caps cells per state with sample-stratified Hamilton allocation,
applies group OE/KD (reusing the perturber's token edits), extracts mean-pooled embeddings from the last layer (or the ISP `emb_layer` with `readout: isp`; batch ≤2 for 4096,
≤8 for 2048), fits PCA→UMAP (or cuML/umap-learn), and writes `*_embs.npy`, `umap_readout.json`, `per_cell_isp_shift.csv`
(`shift_l2`, `shift_toward_{end}`, `cos_shift_toward_{end}`, UMAP before/after) and `umap_{start}_vs_{end}_isp_{label}.png`.

### 日本語

ファインチューニング済みモデルを fp32（`umap.readout: isp` なら bf16）で読み、遺伝子トークンを解決し、サンプルで層別した Hamilton 配分で状態ごとの細胞数を制限します。
グループの OE/KD を適用し（perturber のトークン編集を再利用）、最終層（`readout: isp` なら ISP の `emb_layer`）の平均埋め込みを取り出し（バッチは 4096 なら ≤2、2048 なら ≤8）、
PCA→UMAP（または cuML / umap-learn）で投影します。`*_embs.npy`、`umap_readout.json`、`per_cell_isp_shift.csv`（`shift_l2`、`shift_toward_{end}`、`cos_shift_toward_{end}`、UMAP の前後座標）、
`umap_{start}_vs_{end}_isp_{label}.png` を書きます。
