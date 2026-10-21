### English

`download_build_assets.sh` (profiles; also called by `scripts/macos/setup.sh`), `download_mouse_geneformer.sh` (gdown + `.bin`→safetensors + HF dicts), `download_human_geneformer_v2_104m.sh` (HF), `download_mouse_human_orthologs.sh` (pinned Ensembl 116 tables, SHA-checked; live BioMart only with `ORTHOLOG_REFRESH=1`), `download_drosophila_orthologs.sh` (BioMart), `download_gene_sets.sh` (Enrichr GMT libraries for rank enrichment, SHA-checked; files that already match are kept without network access). V2-316M is never baked.

### 日本語

`download_build_assets.sh`（プロファイル別。`scripts/macos/setup.sh` からも呼ばれる）、`download_mouse_geneformer.sh`（gdown + `.bin`→safetensors + HF の辞書）、`download_human_geneformer_v2_104m.sh`（HF）、`download_mouse_human_orthologs.sh`（固定した Ensembl 116 の表を SHA で確認。BioMart に接続するのは `ORTHOLOG_REFRESH=1` のときだけ）、`download_drosophila_orthologs.sh`（BioMart）、`download_gene_sets.sh`（rank enrichment 用の Enrichr の GMT ライブラリを SHA で確認。すでに一致するファイルがあれば接続しない）。V2-316M は焼き込まれません。
