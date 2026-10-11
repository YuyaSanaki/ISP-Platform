### English

Weights under `models/` (`mouse-Geneformer`, `mouse-Geneformer-12L-E20`, `human-Geneformer-V2-104M`, optional
`human-Geneformer-V2-316M`) and dictionaries under `core/geneformer/dicts/` (mouse, human gc104M, orthologs, cell-type
panels `isp_expression_v1/v2.json`, gene brief function). Mostly git-ignored; baked into the image at
`/opt/geneformer-assets` and seeded into `/app` on container start.

### 日本語

重みは `models/`（`mouse-Geneformer`、`mouse-Geneformer-12L-E20`、`human-Geneformer-V2-104M`、任意で `human-Geneformer-V2-316M`）、
辞書は `core/geneformer/dicts/`（マウス、ヒト gc104M、オルソログ、細胞種パネル `isp_expression_v1/v2.json`、遺伝子機能の短い説明）。
大半は git 管理外で、イメージの `/opt/geneformer-assets` に焼き込まれ、コンテナ起動時に `/app` へコピーされます。
