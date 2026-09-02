# Architecture — monorepo boundaries

This repository is a **monorepo**: Geneformer core and the Streamlit WebUI live in one tree for maintenance, with **clear package boundaries** (same layout as [Mouse-Geneformer-WebUI](https://github.com/YuyaSanaki/Mouse-Geneformer-WebUI)).

Physical split (API service, separate deployables) can wait until job execution needs queueing, multi-tenant isolation, or a network API.

---

## Layout

| Path | Role |
|------|------|
| [`core/`](../core/) | Geneformer library, CLI runners (`run_*.py`), `pipeline_lib`, default YAML under `core/config/` |
| [`webui/`](../webui/) | Streamlit app (`streamlit_app/`), upload helpers, remote URL import |
| [`contracts/`](../contracts/) | Shared rules used by both sides without pulling in job logic — today: [`data_input_layout.py`](../contracts/data_input_layout.py) |
| [`docs/`](../docs/) | Operator docs |
| [`scripts/`](../scripts/) | Download helpers, smoke matrix, entrypoint |
| [`models/`](../models/), [`data/`](../data/), [`output/`](../output/) | Weights / inputs / run outputs (not Python packages) |

`PYTHONPATH` in Docker is `/app/core:/app/contracts:/app/webui` so `import geneformer`, `import data_input_layout`, and `import streamlit_upload` resolve.

---

## WebUI → core contract: subprocess + YAML only

The WebUI must **not** import core runners or `geneformer` to execute jobs. Allowed coupling:

1. **Write** a run-scoped YAML file (e.g. `{WEBUI_WORKSPACE}/runs/<id>/config.yaml`).
2. **Spawn** a subprocess whose argv is a documented core entrypoint + `--config <that file>`.
3. **Observe** exit code, stdout/stderr log, and filesystem outputs under paths declared in the YAML.

Allowed entrypoints (Compose services use the same scripts):

| Run type | Entrypoint | Default template |
|----------|------------|------------------|
| Pipeline (E2E) | `python3 core/run_pipeline.py --config …` | `core/config/pipeline.yaml` |
| ISP UMAP | `python3 core/run_isp_umap.py --config …` | `core/config/isp_umap.yaml` |
| Sequential ISP | `python3 core/run_sequential_isp.py --config …` | `core/config/sequential_isp.yaml` |
| Tokenize (CLI) | `python3 core/execute_tokenizer_pipeline.py` | `TOKENIZE_CONFIG` → `core/config/tokenize.yaml` |
| Fine-tune (CLI) | `python3 core/run_finetune.py --config …` | `core/config/finetune.yaml` |
| ISP (CLI) | `accelerate launch … core/run_isp.py --config …` | `core/config/isp.yaml` |

**Allowed shared import:** `contracts.data_input_layout` (10x study/sample discovery). That module must stay free of geneformer / GPU / runner imports.

**Not allowed:** WebUI importing `pipeline_lib`, `run_pipeline`, `geneformer`, etc. to drive training or ISP.

---

## Core → WebUI

Core must not depend on `webui/`. CLI and Compose one-shot services are the primary automation surface; the WebUI is an optional control panel over the same YAML + scripts.
