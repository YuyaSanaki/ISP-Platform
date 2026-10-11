### English

- It is a sequential simulation, not a time course; a step is not a unit of time.
- Compare only chains with the same number of steps; error grows with each feedback event (`ctrl_reference: start`).
- Removed keys (`feedback_every_step`, `feedback_after_step`, `feedback_after_last_step`, `multi_step`) stop the run with an error (`multistep.check_feedback_config`).
- `state_feedback.placebo_contrast` is applied only by `run_state_feedback_stability.py`; `run_state_feedback_isp.py` trains and reranks with Δh itself and prints a note when the key is on.

### 日本語

- 逐次的なシミュレーションであって時間経過ではありません。1ステップは時間の単位ではありません。
- 比較はステップ数が同じ連鎖どうしに限ります。フィードバックのたびに誤差が積み上がります（`ctrl_reference: start`）。
- 廃止されたキー（`feedback_every_step`、`feedback_after_step`、`feedback_after_last_step`、`multi_step`）があると、エラーで止まります（`multistep.check_feedback_config`）。
- `state_feedback.placebo_contrast` を使うのは `run_state_feedback_stability.py` だけです。`run_state_feedback_isp.py` は Δh そのもので学習・並べ替えを行い、このキーが有効なときは注意を表示します。
