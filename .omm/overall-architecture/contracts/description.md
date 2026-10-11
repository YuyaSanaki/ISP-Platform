### English

`contracts/data_input_layout.py`: rules shared by core and Web UI without importing GPU or job code. Discovers 10x sample
folders (`matrix/barcodes/features`, optionally under `filtered_feature_bc_matrix/`), parses `Time-State-Suffix` folder
names (state = 2nd hyphen field) and produces diagnostics.

### 日本語

`contracts/data_input_layout.py`: core と Web UI が共有するルールで、GPU やジョブのコードは import しません。
10x のサンプルフォルダ（`matrix/barcodes/features`、`filtered_feature_bc_matrix/` 配下も可）を探し、
`Time-State-Suffix` 形式のフォルダ名を解析し（状態はハイフン区切りの2番目）、エラー時の診断を出します。
