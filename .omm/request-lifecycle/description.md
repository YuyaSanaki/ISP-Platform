### English

How a Web UI job flows: study ingest → run YAML → subprocess → polling → outputs, plus the ortholog approval loop. The Web UI never imports core runners; it writes `runs/<id>/config.yaml` and runs `python3 core/run_*.py --config …`.

### 日本語

Web UI のジョブの流れです。study の取り込み → 実行用 YAML → サブプロセス → 状態の確認 → 結果、それにオルソログ承認のループが加わります。Web UI は core のランナーを import せず、`runs/<id>/config.yaml` を書いて `python3 core/run_*.py --config …` を実行します。
