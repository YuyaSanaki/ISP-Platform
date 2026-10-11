### English

Stage topology of an analysis. `run_pipeline.py` runs Tokenize → Fine-tune → ISP → TOP1 ISP UMAP as subprocesses inside
one run directory. ISP UMAP (gene list) and State-feedback ISP are separate run types that reuse a finished pipeline run.
FT batch calibration is an optional one-off before the first pipeline on a GPU.

### 日本語

解析のステージ構成です。`run_pipeline.py` が1つの実行フォルダの中で、トークン化 → ファインチューニング → ISP → TOP1 遺伝子の ISP UMAP
をサブプロセスとして順に動かします。ISP UMAP（遺伝子リスト指定）と State-feedback ISP は別の Run type で、完了済みのパイプライン実行を再利用します。
FT batch calibration は、その GPU で最初にパイプラインを回す前に一度だけ行う任意の手順です。
