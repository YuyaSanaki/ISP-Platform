### English

Reranks see only the first `model_input_size` genes. After a feedback event, genes beyond that window are kept in their original order (`keep_tail` in `replace_input_ids`), so a long encoding is no longer permanently shortened. Pinning still covers OE genes only (KD genes are not pinned). `null_feedback` applies no pin by design: it has no perturbation, so there is nothing to pin; it checks that the decoder alone does not reorder.

### 日本語

並べ替えが見るのは先頭の `model_input_size` 個の遺伝子だけです。feedback のあと、その窓を超えた遺伝子は元の順のまま残します（`replace_input_ids` の `keep_tail`）。そのため長いエンコーディングが恒久的に短くなることはありません。先頭固定の対象は今も OE した遺伝子だけです（KD した遺伝子は固定しません）。`null_feedback` は意図して固定しません。摂動がないので固定する対象がなく、デコーダー単体が並べ替えないことの確認だからです。
