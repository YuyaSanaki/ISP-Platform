### English

`maayanlab.cloud/Enrichr/geneSetLibrary`: `scripts/download_gene_sets.sh` fetches the GMT libraries for rank enrichment (Reactome 2024, GO BP/MF/CC 2025, KEGG 2021 Human, KEGG 2019 Mouse) once, by library name only, and checks them against pinned `SHA256SUMS`. Enrichment itself runs locally; gene lists are never sent.

### 日本語

`maayanlab.cloud/Enrichr/geneSetLibrary`: `scripts/download_gene_sets.sh` が rank enrichment 用の GMT ライブラリ（Reactome 2024、GO BP/MF/CC 2025、KEGG 2021 Human、KEGG 2019 Mouse）をライブラリ名だけで一度取得し、固定した `SHA256SUMS` で確認します。濃縮解析は手元で計算し、遺伝子リストは送りません。
