### English

Pipeline: `_prepare_pipeline_yaml_for_run` validates study, samples, start ≠ end states, overlay and gate genes, then `_build_patched_pipeline_yaml` patches `data.*`, `perturbation.*`, `species.*`, `runtime.*`, `stages.*`. ISP UMAP / State-feedback ISP: built from a past run's `stage_configs/isp.yaml`. Templates come from `core/config/`.

### 日本語

Pipeline: `_prepare_pipeline_yaml_for_run` が study、サンプル、start ≠ end の状態、オーバーレイ、ゲート対象遺伝子を確認し、`_build_patched_pipeline_yaml` が `data.*`、`perturbation.*`、`species.*`、`runtime.*`、`stages.*` を書き換えます。ISP UMAP / State-feedback ISP: 過去の実行の `stage_configs/isp.yaml` から組み立てます。テンプレートは `core/config/` から読みます。
