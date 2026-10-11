### English

`TranscriptomeTokenizer`: vocabulary = intersection of median and token dicts. Each cell is normalised to 10,000 counts,
divided by gene medians, non-zero genes sorted descending and truncated to `max_input_size` (rank-value encoding).
Writes `{prefix}_{n}.dataset` with `input_ids`, `length` and metadata columns. Cross-species input is read from loom in
512-cell chunks and remapped per chunk.

### 日本語

`TranscriptomeTokenizer`: 語彙は中央値辞書とトークン辞書の共通部分です。各細胞を総カウント 10,000 に正規化し、
遺伝子ごとの中央値で割り、0 でない遺伝子を降順に並べて `max_input_size` で切ります（rank-value encoding）。
`input_ids`、`length`、メタデータ列を持つ `{prefix}_{n}.dataset` を書きます。種をまたぐ場合は loom を 512 細胞ずつ読み、塊ごとに変換します。
