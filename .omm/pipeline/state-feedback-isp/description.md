### English

State-feedback multi-step ISP. Steps (OE/KD gene groups) are applied in order on rank tokens; after **every** step the
per-gene hidden-state change Δh is decoded into a rank displacement and the genes are reordered before the next step.
Six conditions run on the same cells: `no_feedback` (baseline), `norm`, `delta_mlm`, `linear_deltarank` (the method),
`oracle` (ceiling), `null_feedback` (safety check). Primary verdict: direction fidelity on held-out genes. Biological
claims require the specific gain over matched placebo chains from the stability runner.

### 日本語

State-feedback 多段階 ISP です。ステップ（OE/KD の遺伝子グループ）を rank トークンに順番に適用し、**毎ステップ後**に遺伝子ごとの隠れ状態の変化 Δh を
rank の移動量に変換して、次のステップの前に遺伝子を並べ替えます。同じ細胞で6条件を走らせます: `no_feedback`（基準）、`norm`、`delta_mlm`、
`linear_deltarank`（本手法）、`oracle`（上限の目安）、`null_feedback`（安全確認）。主な判定は、学習に使わなかった遺伝子での方向の一致度（direction fidelity）です。
生物学的な主張には、安定性ランナーで求めた、対応づけたプラセボ連鎖を上回る特異的な利得が必要です。
