### English

`run_state_feedback_isp.py` never reads `state_feedback.placebo_contrast` and never passes `contrast_subs` to `_build_decoder`, so its `linear_deltarank` always uses plain Δh. Only `run_state_feedback_stability.py` applies the placebo contrast. `docs/state_feedback_isp.md` §2.2 says every feedback event uses it when enabled (the template has `enabled: true`).

### 日本語

`run_state_feedback_isp.py` は `state_feedback.placebo_contrast` を読まず、`_build_decoder` に `contrast_subs` も渡さないため、`linear_deltarank` は常にそのままの Δh を使います。プラセボ差し引きを使うのは `run_state_feedback_stability.py` だけです。一方 `docs/state_feedback_isp.md` §2.2 には、有効時はすべてのフィードバックで使うと書かれています（テンプレートは `enabled: true`）。
