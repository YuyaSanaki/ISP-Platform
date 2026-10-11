### English

`run_isp.main`: reads `paths.dataset`, `paths.geneformer_model` (`CellClassifier`), `perturbation.{state_key, start_state,
end_state, alt_states, genes_to_perturb, type}`. Empty `genes_to_perturb` → genome-wide (can take many hours); a list →
group mode with names converted via the ortholog-aware resolver. Forces `max_input_size` from the backend. Output root
`isp_{time}Z/` (in E2E: the pipeline run dir via `--no-output-*-subdir`). Writes `isp_config_used.yaml`,
`isp_run_metadata.yaml`, `isp_run.log`. In group mode it can trigger ISP UMAP unless `--skip-isp-umap`.

### 日本語

`run_isp.main`: `paths.dataset`、`paths.geneformer_model`（`CellClassifier`）、`perturbation.{state_key, start_state, end_state, alt_states, genes_to_perturb, type}` を読みます。
`genes_to_perturb` が空ならゲノムワイド（何時間もかかることがある）、リストならグループモードで、遺伝子名はオルソログを考慮して変換されます。
`max_input_size` はバックエンドの値で上書きします。出力先は `isp_{time}Z/`（E2E では `--no-output-*-subdir` によりパイプラインの実行フォルダ）。
`isp_config_used.yaml`、`isp_run_metadata.yaml`、`isp_run.log` を書きます。グループモードでは、`--skip-isp-umap` がなければ ISP UMAP も起動します。
