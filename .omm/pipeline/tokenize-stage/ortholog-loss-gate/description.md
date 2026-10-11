### English

Runs when an `ortholog_audit` config exists (or `ortholog_loss_gate: true`). Writes `conversion_gene_detail.tsv`,
`critical_gene_audit.tsv`, `ortholog_loss_summary.json`, `mapping_provenance.json`; on Block also
`ortholog_approval_request.yaml` and the run status `blocked_ortholog_loss_gate`. See
`overall-architecture/geneformer-lib/ortholog-loss-gate` for the verdict rules and `request-lifecycle/approval-card` for approval.

### 日本語

`ortholog_audit` の設定があるとき（または `ortholog_loss_gate: true`）に実行されます。`conversion_gene_detail.tsv`、`critical_gene_audit.tsv`、
`ortholog_loss_summary.json`、`mapping_provenance.json` を書き、Block の場合は `ortholog_approval_request.yaml` と実行状態 `blocked_ortholog_loss_gate` も残します。
判定ルールは `overall-architecture/geneformer-lib/ortholog-loss-gate`、承認の流れは `request-lifecycle/approval-card` を参照してください。
