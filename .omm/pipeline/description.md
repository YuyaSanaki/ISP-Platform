### English

Stage topology of an analysis. `run_pipeline.py` runs Tokenize → Fine-tune → ISP → ISP UMAP as subprocesses inside one
run directory. The E2E ISP UMAP projects the TOP1 significant gene (genome-wide ISP) or the listed genes perturbed together
(gene list). The ISP UMAP run type and State-feedback ISP are separate run types that reuse a finished pipeline run.
FT batch calibration is an optional one-off before the first pipeline on a GPU.

### 日本語

解析のステージ構成です。`run_pipeline.py` が1つの実行フォルダの中で、トークン化 → ファインチューニング → ISP → ISP UMAP
をサブプロセスとして順に動かします。E2E の ISP UMAP は、ゲノムワイド ISP では有意な TOP1 遺伝子を、遺伝子リスト指定ではリストの遺伝子をまとめて摂動したものを投影します。
ISP UMAP の Run type と State-feedback ISP は別の Run type で、完了済みのパイプライン実行を再利用します。
FT batch calibration は、その GPU で最初にパイプラインを回す前に一度だけ行う任意の手順です。
