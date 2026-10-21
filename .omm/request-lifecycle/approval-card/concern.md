### English

The project default overlay (`analysis/ortholog_policy/v1`) and audit manifest (`analysis/analysis_manifest.yaml`) are intentionally not shipped: a curated bridge is a curation decision for each analysis. The card says so, names the overlay it uses and its source, and its re-run message names the YAML keys it sets (`stages.tokenize.tokenizer.ortholog_approval_record`, `species.ortholog_curated_overlay`). A curated-bridge approval therefore needs a user-supplied overlay.

### 日本語

プロジェクトの既定のオーバーレイ（`analysis/ortholog_policy/v1`）と監査マニフェスト（`analysis/analysis_manifest.yaml`）は意図して同梱していません。curated bridge は解析ごとのキュレーションの判断だからです。カードはそのことを示し、使うオーバーレイとその出どころを表示し、再実行のメッセージでは設定する YAML のキー（`stages.tokenize.tokenizer.ortholog_approval_record`、`species.ortholog_curated_overlay`）を示します。そのため curated bridge の承認には、利用者が用意したオーバーレイが必要です。
