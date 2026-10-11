### English

`iter_gene_state_pairs`: forward reference and perturbed encodings in chunks, align genes by **token identity** (not position), Δh = h_pert − h_ctrl (genes missing in the reference get 0). Also MLM self-logits for `delta_mlm` via the pretrained model.

### 日本語

`iter_gene_state_pairs`: 基準と摂動後のエンコーディングを塊ごとに forward し、遺伝子を位置ではなく**トークンの一致**で揃えて、Δh = h_pert − h_ctrl を求めます（基準にない遺伝子は 0）。`delta_mlm` 用に、事前学習モデルで MLM の自己ロジットも計算します。
