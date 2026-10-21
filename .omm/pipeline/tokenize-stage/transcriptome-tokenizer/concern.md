### English

Same-species path: `max_cells` is a shard-split threshold, so data can spill into `_1.dataset`…; the pipeline concatenates the shards into `{study}_merged.dataset` and uses all of them. Cross-species path: `max_cells` still truncates each file instead.

### 日本語

同種の経路では `max_cells` が分割のしきい値なので、データが `_1.dataset` 以降に分かれることがあります。パイプラインはそれらを `{study}_merged.dataset` に結合してすべて使います。種をまたぐ経路では、`max_cells` は今も各ファイルの切り詰めとして働きます。
