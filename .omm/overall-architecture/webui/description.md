### English

Streamlit control panel (`webui/streamlit_app/app.py`, ~5,500 lines) plus upload (`streamlit_upload.py`), URL import
(`streamlit_remote_data.py`), ortholog approval card and dropped-gene panel. Run types: Pipeline (E2E), FT batch size
(calibrate), ISP UMAP, State-feedback ISP. State-feedback ISP has two modes: **Placebo-corrected (paper method)**
(default, `run_state_feedback_stability.py`; Outputs show τ, S1–S4 and the placebo genes with CSV downloads) and
**Decoder diagnostics (no placebo correction)** (`run_state_feedback_isp.py`). Pipeline (E2E) and the placebo-corrected
mode have a **Cluster / cell-type / enrichment analysis** expander (`postprocess.enabled`, `rank_enrichment`); rank
enrichment figures and tables are shown with the run. The fine-tune `train_batch_size` field accepts 4–8 only.
**Stop job** sits under **Logs & status**. See the `request-lifecycle` perspective for the job flow.

### 日本語

Streamlit の操作画面です（`webui/streamlit_app/app.py`、約5,500行）。ほかに zip アップロード（`streamlit_upload.py`）、
URL からの取り込み（`streamlit_remote_data.py`）、オルソログ承認カード、変換で落ちた遺伝子の表示があります。
Run type は Pipeline (E2E)、FT batch size (calibrate)、ISP UMAP、State-feedback ISP の4つ。State-feedback ISP にはモードが2つあります:
**Placebo-corrected (paper method)**（既定、`run_state_feedback_stability.py`。Outputs に τ、S1〜S4、プラセボ遺伝子を表示し CSV で保存できる）と
**Decoder diagnostics (no placebo correction)**（`run_state_feedback_isp.py`）。Pipeline (E2E) と Placebo-corrected のモードには
**Cluster / cell-type / enrichment analysis** の欄があり（`postprocess.enabled`、`rank_enrichment`）、rank enrichment の図と表は実行結果と一緒に表示されます。
ファインチューニングの `train_batch_size` の欄は 4〜8 だけを受け付けます。**Stop job** は **Logs & status** の下にあります。ジョブの流れは `request-lifecycle` を参照してください。
