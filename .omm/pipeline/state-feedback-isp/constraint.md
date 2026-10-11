### English

- It is a sequential simulation, not a time course; a step is not a unit of time.
- Compare only chains with the same number of steps; error grows with each feedback event (`ctrl_reference: start`).
- Removed keys (`feedback_every_step`, `feedback_after_step`, `feedback_after_last_step`, `multi_step`) stop the run with an error (`multistep.check_feedback_config`).

### 日本語

- 逐次的なシミュレーションであって時間経過ではありません。1ステップは時間の単位ではありません。
- 比較はステップ数が同じ連鎖どうしに限ります。フィードバックのたびに誤差が積み上がります（`ctrl_reference: start`）。
- 廃止されたキー（`feedback_every_step`、`feedback_after_step`、`feedback_after_last_step`、`multi_step`）があると、エラーで止まります（`multistep.check_feedback_config`）。
