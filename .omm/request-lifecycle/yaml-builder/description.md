### English

Pipeline: `_prepare_pipeline_yaml_for_run` validates study, samples, start ≠ end states, overlay and gate genes, then `_build_patched_pipeline_yaml` patches `data.*`, `perturbation.*`, `species.*`, `runtime.*`, `stages.*`. ISP UMAP / State-feedback ISP: built from a past run's `stage_configs/isp.yaml`. State-feedback ISP adds `stability.*` and `placebo_contrast.enabled: true` (default mode; correction draw seed must differ from the comparison draw seeds) or `conditions`, optional `specificity` and `placebo_contrast.enabled: false` (Decoder diagnostics). The YAML is rebuilt from the form on **Run job**. Templates come from `core/config/`.

### 日本語

Pipeline: `_prepare_pipeline_yaml_for_run` が study、サンプル、start ≠ end の状態、オーバーレイ、ゲート対象遺伝子を確認し、`_build_patched_pipeline_yaml` が `data.*`、`perturbation.*`、`species.*`、`runtime.*`、`stages.*` を書き換えます。ISP UMAP / State-feedback ISP: 過去の実行の `stage_configs/isp.yaml` から組み立てます。State-feedback ISP では、既定のモードで `stability.*` と `placebo_contrast.enabled: true` を加え（補正用プラセボの抽選シードは比較用と違う値にする）、Decoder diagnostics では `conditions`、任意で `specificity`、`placebo_contrast.enabled: false` を加えます。YAML は **Run job** のときにフォームから作り直します。テンプレートは `core/config/` から読みます。
