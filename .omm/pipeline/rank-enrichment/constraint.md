### English

- Gene lists never leave the machine: the libraries are local GMT files fetched once by `scripts/download_gene_sets.sh` (only the library name goes to Enrichr) and checked against pinned `SHA256SUMS`; a changed library is rejected unless `GENE_SETS_ALLOW_MISMATCH=1`.
- Off by default and under one switch with cluster / cell-type analysis (`postprocess.enabled`, `core/postprocess_switch.py`); it never changes ISP scores or τ.
- Group and per-cell ISP outputs are skipped with a message; only per-gene screens and the State-feedback stability runner are supported.

### 日本語

- 遺伝子リストは手元から出ません。ライブラリは `scripts/download_gene_sets.sh` で一度だけ取得した GMT ファイルで（Enrichr に送るのはライブラリ名だけ）、固定した `SHA256SUMS` で確認します。内容が変わったライブラリは `GENE_SETS_ALLOW_MISMATCH=1` のとき以外は受け付けません。
- 既定はオフで、クラスタ / 細胞種解析と同じスイッチ（`postprocess.enabled`、`core/postprocess_switch.py`）で動きます。ISP のスコアと τ は変えません。
- グループと細胞ごとの ISP の出力はメッセージを出して飛ばします。対応するのは遺伝子ごとのスクリーニングと State-feedback の stability runner だけです。
