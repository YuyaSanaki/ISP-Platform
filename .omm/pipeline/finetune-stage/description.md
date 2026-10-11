### English

`run_finetune.py`: `BertForSequenceClassification.from_pretrained(..., ignore_mismatched_sizes=True)` trained with HF
`Trainer` and `DataCollatorForCellClassification`. Label column (default `disease`) → `label`; rare classes
(≤0.5%) dropped; shuffled; sequential 80/20 split (not stratified); optional `freeze_layers`. Writes
`finetune/all_run1/` (model, `checkpoint-*`, `label_dict.json`, `eval_results.json`, `preds.pkl`), `best_*/`,
optional CLS-embedding UMAP in `figures/`, plus log and provenance.

### 日本語

`run_finetune.py`: `BertForSequenceClassification.from_pretrained(..., ignore_mismatched_sizes=True)` を HF の `Trainer` と
`DataCollatorForCellClassification` で学習します。ラベル列（既定は `disease`）を `label` にし、0.5% 以下の少数クラスを除き、シャッフルして、
前から順に 80/20 に分けます（層別ではない）。`freeze_layers` で下位層を固定することもできます。
`finetune/all_run1/`（モデル、`checkpoint-*`、`label_dict.json`、`eval_results.json`、`preds.pkl`）、`best_*/`、
任意で `figures/` に CLS 埋め込みの UMAP、さらにログと由来情報を書きます。
