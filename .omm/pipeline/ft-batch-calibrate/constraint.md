### English

Fine-tune batch size changes optimizer steps per epoch and therefore the model; changing it within a study makes runs incomparable. With the fixed learning rate and epochs, values outside 4–8 (around the default 6) often change the selected checkpoint and the ISP results (batch 2 vs 6 is a known failure), so the Web UI allows only 4–8. Calibration measures what fits in GPU memory, not what is stable.

### 日本語

ファインチューニングのバッチサイズは1エポックあたりの更新回数、つまり学習結果を変えます。同じ study の途中で変えると実行どうしを比較できなくなります。学習率とエポック数が固定なので、4〜8（既定の 6 の周辺）から外れると選ばれるチェックポイントと ISP の結果が変わりやすく（バッチ 2 と 6 の違いは既知の失敗例）、Web UI は 4〜8 しか受け付けません。calibrate が測るのは GPU メモリに収まる値で、結果が安定する値ではありません。
