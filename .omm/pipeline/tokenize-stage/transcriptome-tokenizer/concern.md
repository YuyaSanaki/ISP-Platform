### English

Same-species path: `max_cells` is a shard-split threshold, so data can spill into `_1.dataset`… which the pipeline never reads. Cross-species path: `max_cells` truncates each file instead.

### 日本語

同種の経路では `max_cells` が分割のしきい値なので、データが `_1.dataset` 以降にこぼれることがあり、パイプラインはそれを読みません。種をまたぐ経路では、`max_cells` は各ファイルの切り詰めとして働きます。
