### English

`InSilicoPerturber` (conventional ISP). Loads the model in bf16, computes one centroid per state (start, goal, alt) from
mean-pooled hidden states at `quant_layers + emb_layer`, then perturbs start-state cells.
**Group mode** (gene list): all genes in one edit; KD deletes tokens (cell must contain every gene), OE moves/inserts them
at the front and trims the tail (length-preserving). **Genome-wide mode** (`genes_to_perturb: []`): one forward pass per
gene per cell. Score: `shift = cos(perturbed, centroid) − cos(original, centroid)` per state.

### 日本語

`InSilicoPerturber`（従来型 ISP）。モデルを bf16 で読み込み、`quant_layers + emb_layer` 層の隠れ状態を平均して状態ごと（start、goal、alt）の重心を作り、
start 状態の細胞を摂動します。**グループモード**（遺伝子リスト指定）は全遺伝子を1回で編集します。KD はトークンを削除し（細胞が全遺伝子を含む必要あり）、
OE は先頭へ移動・挿入して末尾を切ります（長さを保つ）。**ゲノムワイドモード**（`genes_to_perturb: []`）は細胞ごと・遺伝子ごとに1回ずつ forward します。
スコアは状態ごとに `shift = cos(摂動後, 重心) − cos(元, 重心)`。
