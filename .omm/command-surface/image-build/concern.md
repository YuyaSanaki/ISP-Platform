### English

Docker, macOS and H100 now use Python 3.12 and install the same `requirements-pins.txt` (`transformers<5`, `datasets==5.0.1`). The other `requirements.txt` pins are still stripped (they date from Python 3.8), so transitive versions can drift between builds. Every run records Python and key package versions under `environment` in its metadata, so a result can be traced to its stack.

### 日本語

Docker、macOS、H100 はいずれも Python 3.12 を使い、同じ `requirements-pins.txt`（`transformers<5`、`datasets==5.0.1`）を入れるようになりました。`requirements.txt` のその他の固定は外したままです（Python 3.8 時代のもののため）。そのため間接的な依存のバージョンはビルドごとに変わりえます。各実行はメタデータの `environment` に Python と主要パッケージのバージョンを記録するので、結果がどの環境のものか追えます。
