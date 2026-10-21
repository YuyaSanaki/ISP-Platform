### English

- The Web UI starts jobs only as subprocesses driven by YAML. From core it may import only `data_input_layout`, `platform_paths`, `geneformer.gene_converter` and `geneformer.ortholog_loss_gate` (display, path mapping and approval records). `tests/test_webui_core_boundary.py` enforces this list.
- `geneformer/__init__.py` loads the tokenizer, perturber and pretrainer lazily, so these imports do not pull torch, transformers or datasets into the Web UI process.

### 日本語

- Web UI がジョブを始めるのは、YAML で動かすサブプロセスとしてだけです。core から import してよいのは `data_input_layout`、`platform_paths`、`geneformer.gene_converter`、`geneformer.ortholog_loss_gate` だけです（表示、パスの読み替え、承認記録のため）。`tests/test_webui_core_boundary.py` がこの一覧を強制します。
- `geneformer/__init__.py` は tokenizer、perturber、pretrainer を遅延読み込みするので、これらの import で Web UI のプロセスに torch、transformers、datasets は読み込まれません。
