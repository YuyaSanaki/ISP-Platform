### English

`TranscriptomeTokenizer.tokenize_data`: rank-value encoding per cell, truncated to the backend `max_input_size` (2048 mouse / 4096 human). Cross-species input is remapped in 512-cell loom chunks. Output: `tokenized_dataset/{study}_{n}.dataset`.

### 日本語

`TranscriptomeTokenizer.tokenize_data`: 細胞ごとに rank-value encoding を作り、バックエンドの `max_input_size`（マウス 2048 / ヒト 4096）で切ります。種をまたぐ入力は loom を 512 細胞ずつ変換します。出力は `tokenized_dataset/{study}_{n}.dataset`。
