# contracts/

Shared rules used by **both** `core/` and `webui/` without pulling job or GPU logic.

Today: `data_input_layout.py` (10x study/sample discovery).

Do not import `geneformer`, runners, or Streamlit here. See `docs/architecture.md`.
