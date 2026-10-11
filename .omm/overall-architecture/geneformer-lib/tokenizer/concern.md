### English

- `tokenize_data` does `continue` when a round yields zero cells without advancing the file counter, so it can loop forever.
- No `<cls>`/`<eos>` tokens are added; upstream Geneformer V2 uses them, so human V2 inputs may differ from its pretraining format (needs verification).

### 日本語

- `tokenize_data` は、ある回で細胞が0個だとファイル番号を進めずに `continue` するため、無限ループになり得ます。
- `<cls>`/`<eos>` トークンを付けていません。本家 Geneformer V2 はこれらを使うので、ヒト V2 への入力が事前学習時の形式とずれている可能性があります（要確認）。
