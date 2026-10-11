### English

Streamlit control panel (`webui/streamlit_app/app.py`, ~4,800 lines) plus upload (`streamlit_upload.py`), URL import
(`streamlit_remote_data.py`), ortholog approval card and dropped-gene panel. Run types: Pipeline (E2E), FT batch size
(calibrate), ISP UMAP, State-feedback ISP. See the `request-lifecycle` perspective for the job flow.

### 日本語

Streamlit の操作画面です（`webui/streamlit_app/app.py`、約4,800行）。ほかに zip アップロード（`streamlit_upload.py`）、
URL からの取り込み（`streamlit_remote_data.py`）、オルソログ承認カード、変換で落ちた遺伝子の表示があります。
Run type は Pipeline (E2E)、FT batch size (calibrate)、ISP UMAP、State-feedback ISP の4つ。ジョブの流れは `request-lifecycle` を参照してください。
