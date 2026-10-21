### English

- Backends: Mouse Geneformer (base 6L, 12L-E20; seq 2048) and Human Geneformer (V2-104M default, V2-316M; seq 4096).
- Primary test host: DGX Spark (aarch64); H100 (x86_64) native conda path is also supported. Apple silicon Macs run natively on MPS (checked on an M1 Max, macOS 26); fp16 fine-tuning stays CUDA-only.
- Architecture rule (`docs/architecture.md`): Web UI → core coupling is subprocess + YAML only; core never imports `webui/`.

### 日本語

- バックエンド: Mouse Geneformer（base 6層、12L-E20、系列長 2048）と Human Geneformer（V2-104M が既定、V2-316M、系列長 4096）。
- 主な検証環境は DGX Spark（aarch64）。H100（x86_64）の conda によるネイティブ実行にも対応。Apple silicon の Mac では MPS でネイティブに動きます（M1 Max、macOS 26 で確認）。fp16 のファインチューニングは CUDA だけです。
- 設計ルール（`docs/architecture.md`）: Web UI と core のつながりは「サブプロセス + YAML」だけ。core は `webui/` を import しない。
