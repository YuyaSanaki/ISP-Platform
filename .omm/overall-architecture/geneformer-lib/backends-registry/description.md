### English

`parse_species_config` validates `species.*` (model_organism mouse|human|drosophila, model mouse_geneformer|human_geneformer,
mouse_variant base|12l_e20, human_variant v2_104m|v2_316m, ortholog_policy one2one|best_of_n|legacy_sum).
`BACKENDS` maps each variant to its model dir under `GENEFORMER_MODELS_ROOT`, its token/median dicts and `max_input_size`
(2048 mouse, 4096 human). `needs_gene_conversion` is true when data species ≠ model species. `species_context.py` adds
default batch sizes (fine-tune 2 vs 6, ISP 25 vs 100 for long vs short sequences).

### 日本語

`parse_species_config` が `species.*` を検証します（model_organism は mouse|human|drosophila、model は mouse_geneformer|human_geneformer、
mouse_variant は base|12l_e20、human_variant は v2_104m|v2_316m、ortholog_policy は one2one|best_of_n|legacy_sum）。
`BACKENDS` は各バリアントを、`GENEFORMER_MODELS_ROOT` 配下のモデルフォルダ、トークン辞書・中央値辞書、`max_input_size`
（マウス 2048、ヒト 4096）に対応づけます。データの種とモデルの種が違うと `needs_gene_conversion` が真になります。
`species_context.py` は既定のバッチサイズ（系列が長いか短いかで、ファインチューニング 2 か 6、ISP 25 か 100）を足します。
