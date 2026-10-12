### English

Streamlit control panel (`webui/streamlit_app/app.py`, ~5,200 lines) plus upload (`streamlit_upload.py`), URL import
(`streamlit_remote_data.py`), ortholog approval card and dropped-gene panel. Run types: Pipeline (E2E), FT batch size
(calibrate), ISP UMAP, State-feedback ISP. State-feedback ISP has two modes: **Placebo-corrected (paper method)**
(default, `run_state_feedback_stability.py`; Outputs show τ, S1–S4 and the placebo genes with CSV downloads) and
**Decoder diagnostics (no placebo correction)** (`run_state_feedback_isp.py`). See the `request-lifecycle` perspective
for the job flow.

### 日本語

Streamlit の操作画面です（`webui/streamlit_app/app.py`、約5,200行）。ほかに zip アップロード（`streamlit_upload.py`）、
URL からの取り込み（`streamlit_remote_data.py`）、オルソログ承認カード、変換で落ちた遺伝子の表示があります。
Run type は Pipeline (E2E)、FT batch size (calibrate)、ISP UMAP、State-feedback ISP の4つ。State-feedback ISP にはモードが2つあります:
**Placebo-corrected (paper method)**（既定、`run_state_feedback_stability.py`。Outputs に τ、S1〜S4、プラセボ遺伝子を表示し CSV で保存できる）と
**Decoder diagnostics (no placebo correction)**（`run_state_feedback_isp.py`）。ジョブの流れは `request-lifecycle` を参照してください。
