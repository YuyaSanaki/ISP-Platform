### English

- It is a sequential simulation, not a time course; a step is not a unit of time.
- Compare only chains with the same number of steps; error grows with each feedback event (`ctrl_reference: start`).
- Removed keys (`feedback_every_step`, `feedback_after_step`, `feedback_after_last_step`, `multi_step`) stop the run with an error (`multistep.check_feedback_config`).
- `state_feedback.placebo_contrast` is applied only by `run_state_feedback_stability.py`; `run_state_feedback_isp.py` trains and reranks with Δh itself and prints a note when the key is on. The Web UI writes `enabled: true` in its default mode (stability runner) and `enabled: false` in Decoder diagnostics.
- The correction placebo draw seed (`placebo_contrast.estimation_draw_seed`) must differ from the comparison draw seeds (`stability.random_seeds`); the runner and the Web UI stop otherwise. Defaults 1 and [0] differ.

### 日本語

- 逐次的なシミュレーションであって時間経過ではありません。1ステップは時間の単位ではありません。
- 比較はステップ数が同じ連鎖どうしに限ります。フィードバックのたびに誤差が積み上がります（`ctrl_reference: start`）。
- 廃止されたキー（`feedback_every_step`、`feedback_after_step`、`feedback_after_last_step`、`multi_step`）があると、エラーで止まります（`multistep.check_feedback_config`）。
- `state_feedback.placebo_contrast` を使うのは `run_state_feedback_stability.py` だけです。`run_state_feedback_isp.py` は Δh そのもので学習・並べ替えを行い、このキーが有効なときは注意を表示します。Web UI は既定のモード（安定性ランナー）で `enabled: true`、Decoder diagnostics で `enabled: false` を書きます。
- 補正用プラセボの抽選シード（`placebo_contrast.estimation_draw_seed`）は、比較用の抽選シード（`stability.random_seeds`）と違う値にします。同じならランナーも Web UI も止まります。既定値の 1 と [0] は異なります。
