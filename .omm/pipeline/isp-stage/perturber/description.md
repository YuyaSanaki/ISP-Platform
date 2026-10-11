### English

See `overall-architecture/geneformer-lib/in-silico-perturber`. Genome-wide mode writes a pickle every 100 cells and resets the dict every 1,000 cells; multi-GPU splits contiguous cell ranges with `_rank{i}` prefixes.

### 日本語

詳細は `overall-architecture/geneformer-lib/in-silico-perturber` を参照してください。ゲノムワイドモードは 100 細胞ごとに pickle を書き、1,000 細胞ごとに辞書をリセットします。複数 GPU では連続した細胞範囲を分担し、ファイル名に `_rank{i}` を付けます。
