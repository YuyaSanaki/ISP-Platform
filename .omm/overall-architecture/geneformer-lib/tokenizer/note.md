### English

- Inputs are a plain rank-value encoding for every backend, Human V2 included: no `<cls>`/`<eos>`, highest rank at position 0 (`docs/tokenization.md` § Input format).
- Fine-tuning trains all layers, so models and ISP see the format they were fine-tuned on. Only the `delta_mlm` comparator of State-feedback ISP uses the pretrained model as is; adding `<cls>`/`<eos>` there leaves direction fidelity unchanged on BBRC n=300.

### 日本語

- 入力はすべてのバックエンド（ヒト V2 も含む）で、特殊トークンを付けない rank-value encoding です。`<cls>`/`<eos>` はなく、位置 0 は最上位の遺伝子です（`docs/tokenization.md` の Input format）。
- ファインチューニングは全層を学習するため、モデルと ISP はファインチューニング時と同じ形式を見ます。事前学習モデルをそのまま使うのは State-feedback ISP の `delta_mlm` 比較だけで、そこに `<cls>`/`<eos>` を付けても BBRC n=300 の direction fidelity は変わりませんでした。
