### English

State-feedback multi-step ISP. Steps (OE/KD gene groups) are applied in order on rank tokens; after **every** step the
per-gene hidden-state change Δh is decoded into a rank displacement and the genes are reordered before the next step.
Two runners. The stability runner is the paper's method: the decoder reads Δh minus the mean Δh of correction placebos,
and the factor-specific effect τ is the feedback gain minus that of matched comparison placebo chains. It is the Web UI
default mode, **Placebo-corrected (paper method)**. The SF runner (Web UI mode **Decoder diagnostics**) checks the decoder
without placebos: six conditions on the same cells, `no_feedback` (baseline), `norm`, `delta_mlm`, `linear_deltarank`
(the method), `oracle` (ceiling), `null_feedback` (safety check), with direction fidelity on held-out genes as its
verdict. Biological claims require τ from the stability runner.

### 日本語

State-feedback 多段階 ISP です。ステップ（OE/KD の遺伝子グループ）を rank トークンに順番に適用し、**毎ステップ後**に遺伝子ごとの隠れ状態の変化 Δh を
rank の移動量に変換して、次のステップの前に遺伝子を並べ替えます。ランナーは2つあります。安定性ランナーが論文の手法で、デコーダーは Δh から補正用プラセボの平均 Δh を
引いたものを読み、因子に特異的な効果 τ は、フィードバックの利得から対応づけた比較用プラセボ連鎖の利得を引いたものです。Web UI の既定モード
**Placebo-corrected (paper method)** はこれを実行します。SF ランナー（Web UI のモード **Decoder diagnostics**）はプラセボなしでデコーダーを確かめます。
同じ細胞で6条件を走らせます: `no_feedback`（基準）、`norm`、`delta_mlm`、`linear_deltarank`（本手法）、`oracle`（上限の目安）、`null_feedback`（安全確認）。
判定は、学習に使わなかった遺伝子での方向の一致度（direction fidelity）です。生物学的な主張には、安定性ランナーで求めた τ が必要です。
