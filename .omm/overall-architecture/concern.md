### English

- The Web UI display helpers (`dropped_genes_ui.py`, `ortholog_approval_ui.py`, overlay resolution in `app.py`) import parts of `geneformer`, which bends the "no geneformer import" rule. Display-only, not job execution.
- `app.py` also imports `core/platform_paths.py` (to map `/app/...` in templates, saved runs and logs on macOS). `docs/architecture.md` lists only `contracts.data_input_layout` as an allowed shared import. Path mapping only, not job execution.

### 日本語

- Web UI の表示用ヘルパー（`dropped_genes_ui.py`、`ortholog_approval_ui.py`、`app.py` のオーバーレイ解決）は `geneformer` の一部を import しており、「geneformer を import しない」ルールから少し外れています。ジョブ実行ではなく表示のためだけです。
- `app.py` は `core/platform_paths.py` も import しています（macOS でテンプレート、保存済みの実行、ログの `/app/...` を読み替えるため）。`docs/architecture.md` が共有の import として認めているのは `contracts.data_input_layout` だけです。パスの読み替えだけで、ジョブ実行には使いません。
