# ISP Platform Web UI

Streamlit control panel for this repository (CLI-equivalent jobs via subprocess + YAML).

- App: `[webui/streamlit_app/app.py](../webui/streamlit_app/app.py)`
- Compose service: `[docker-compose.yml](../docker-compose.yml)` (`platform_webui`)
- Overview: [README § Quick start (Web UI)](../README.md#quick-start-web-ui--pipeline-e2e)

There is **no Jupyter Lab** service in Compose. Use **CLI** (`docker compose run …`) or this Web UI only.

---



## Start

```bash
docker compose build isp-platform   # downloads models into the image by default
docker compose up -d platform_webui
```

Open **[http://localhost:8502](http://localhost:8502)**.

Build needs network (Google Drive, Hugging Face, Ensembl BioMart). Override with
`DOWNLOAD_MODELS=none` / `minimal` / `all` — see [models/README.md](../models/README.md).
The `.:/app` bind mount does **not** hide baked weights: the entrypoint seeds
`/opt/geneformer-assets` → `./models` + `core/geneformer/dicts/` when host paths are empty.

On a **remote GPU server**, forward the port:

```bash
ssh -L 8502:localhost:8502 <user>@<server>
```

Then open **[http://localhost:8502](http://localhost:8502)** on your laptop. See also LAN/Tailscale access if your network allows `http://<server-ip>:8502`.

`WANDB_DISABLED=true` by default. For multi-GPU ISP from the UI, set `ISP_NUM_GPUS` in the environment or `.env`.

**GPU note:** the `platform_webui` service requests **1 GPU** because **Run job** launches tokenize/finetune/ISP as in-container subprocesses that need CUDA. Idle Jupyter is not used.

**Apple silicon Mac:** no Docker. After `bash scripts/macos/setup.sh` (README, macOS), start the Web UI with `bash scripts/macos/webui.sh` and open **[http://localhost:8502](http://localhost:8502)**. Jobs run on the Metal GPU (MPS). Paths shown as `/app/...` in templates are mapped to the checkout.

---



## Environment variables


| Variable                           | Meaning                                                                       |
| ---------------------------------- | ----------------------------------------------------------------------------- |
| `WEBUI_ROOT`                       | Repository root (Compose sets `/app`)                                         |
| `WEBUI_WORKSPACE`                  | Uploads and per-run folders (default `{WEBUI_ROOT}/data/streamlit_workspace`) |
| `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` | Upload cap in megabytes (see `.streamlit/config.toml`)                        |
| `ISP_WEBUI_USERS_FILE`             | Optional login: YAML `users: {name: hash}`; see below                         |
| `ISP_URL_IMPORT_ALLOW_PRIVATE`     | `1` lets **Download from URL** fetch localhost / LAN hosts (default: public only) |

**Login.** The Web UI has no login unless `ISP_WEBUI_USERS_FILE` is set. Without it, run it for one trusted user on localhost or behind a proxy that controls access. To turn login on, create entries with `python webui/webui_auth.py hash <user>` (prompts for the password; stores a PBKDF2 hash), put them in one YAML file and point `ISP_WEBUI_USERS_FILE` at it. Every page then asks for a user name and password, and the ortholog approval card records the logged-in user as `approved_by` and `authenticated_user` instead of a free-text name.

**Download from URL** fetches only public addresses. URLs whose host resolves to localhost, a private LAN range, link-local (cloud metadata) or another non-public address are refused, also after redirects, because the server makes the request. Prepared run zips are written to `{WEBUI_WORKSPACE}/downloads/`; zips over 2 GiB are shown as a server path instead of a download button.


Per **Run job**:

`{WEBUI_WORKSPACE}/runs/<UTC>_<id>/config.yaml` and `console.log`

Uploads:

`{WEBUI_WORKSPACE}/uploads/<session>/<study_name>/`

Remote archives: Web UI **Download from URL**.

After a successful **Pipeline (E2E)** or **ISP UMAP** job, the **Outputs** section on the **Analysis** tab offers **Download figures (.zip)** when figure files exist under `paths.output_root`.

---



## Species / conversion (Web UI)

On **Analysis → Pipeline (E2E)** (also FT calibrate), the **Species / model** block sets what YAML would call `species.`*. Changing a dropdown patches the session YAML immediately.


| UI label                                          | Writes                                    | Meaning                                                                                                                                                              |
| ------------------------------------------------- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Species of the uploaded scRNA-seq**             | `species.model_organism`                  | Organism of gene IDs in the zip (`mouse` / `human` / `drosophila` **(Beta)** — Web UI label: **Drosophila (fruit fly) (Beta)**; not biologically validated)          |
| **Pretrained Geneformer**                         | `species.model`                           | Checkpoint for FT/ISP (`mouse_geneformer` / `human_geneformer`)                                                                                                      |
| **Mouse / Human variant**                         | `species.mouse_variant` / `human_variant` | YAML ids stay `base` / `12l_e20`, `v2_104m` / `v2_316m`; Web UI shows **Base** / **Large** (e.g. Base (6L / ~33M), Large (12L-E20), Base (V2-104M), Large (V2-316M)) |
| **How to map genes when several orthologs exist** | `species.ortholog_policy`                 | Shown **only** when input species ≠ model native species                                                                                                             |
| **Project curated overlay (optional)** | `species.ortholog_curated_overlay` | Cross-species only. Container path to a TSV or to a folder with `curated_bridge_<pair>.tsv` (repository = `/app`). Empty = platform tables only. The run does not start if no table is found for the current pair |
| **Ortholog loss gate (optional)** + **Critical genes** | `stages.tokenize.tokenizer.ortholog_audit_inline` (`critical_sets.web_ui`), `ortholog_loss_gate: true` | Cross-species only. Blocks tokenize if a listed gene is in the input but not mapped; warns if it is absent from the input. At least one gene is required |


**Same-species** (e.g. mouse data + mouse Geneformer): caption says no ortholog conversion; policy control is hidden.

**Cross-species** (e.g. human data + mouse Geneformer): an info banner shows `human → mouse`, and the ortholog policy select appears (default **one2one**). Expand **Which ortholog policy should I pick?** for the three options.

Conversion runs automatically at **tokenize** and **ISP**. After tokenize you get `conversion_report.json` / `conversion_unmapped_genes.tsv` under the run’s tokenized output (see [tokenization.md](tokenization.md)).

**Dropped-gene table (Web UI):** **Output** (and Analysis → Outputs after a Pipeline run) shows whether ortholog conversion dropped genes, which IDs/symbols, why (e.g. one-to-many), and a **brief function** one-liner (curated note when we have one, otherwise NCBI official full name from `gene_brief_function.tsv.gz`). Same-species runs have no conversion report.

There is **no free-form gene-ID field** for ortholog pairs in the UI. Project bridges use a curated overlay file on disk, entered in **Project curated overlay** (the approval card falls back to `analysis/ortholog_policy/v1` when present), not hand-typed Ensembl IDs. The repository ships no project overlay and no `analysis/analysis_manifest.yaml`: a curated bridge and an audit manifest are decisions for each analysis, so supply your own reviewed files. The card says which overlay it uses and where it came from. The **Critical genes** field of the loss gate lists genes to check, not ortholog pairs.

### Worked examples (species selectors)

**A — Native mouse (no conversion)**

1. Upload a mouse 10x study.
2. Input species: **Mouse** · Model: **Mouse Geneformer** · Variant: e.g. **base**.
3. Ortholog policy is unused. Run job → tokenize proceeds without remap.

**B — Human data on mouse Geneformer (strict one2one)**

1. Upload human scRNA-seq.
2. Input species: **Human** · Model: **Mouse Geneformer**.
3. Ortholog mapping appears → leave **one2one**.
4. Run **Pipeline (E2E)**. Ambiguous orthologs are dropped; check `conversion_unmapped_genes.tsv` if a gene of interest is missing from ISP.

**C — Fly → human (Beta — coverage note)**

1. Input species: **Drosophila (fruit fly) (Beta)** · Model: **Human Geneformer** · Variant: **Base (V2-104M)**.
2. Keep **one2one**. Expect lower gene retention than mouse↔human; use the conversion report before interpreting ISP.
3. Fly input is **Beta**: smoke / pipeline paths exist, but results are **not** biologically validated — do not treat as production-ready.

**D — Critical gene Block → approve curated bridge**

Requires an analysis contract that enables the gate (e.g. `analysis/analysis_manifest.yaml` → `ortholog_audit`). Typical path:

1. Cross-species run as in **B** (or human→mouse / mouse→human as your study needs), with audit wired so `ortholog_loss_gate` runs.
2. If a critical gene is **present in the matrix** but dropped by `one2one` (a one-to-many gene that the platform curated table does not restore; POU5F1 and NANOG are restored by default), tokenize **exits non-zero** and writes `ortholog_approval_request.yaml` (`status: pending`).
3. Scroll to **Outputs** → card **Ortholog mapping — approval required** (see next section).
4. Fill **Approved by** / **Reason**, choose **Approve curated bridge**, **Record decision**.
5. UI writes `approval_record.yaml` and starts a **new** Pipeline with approval + overlay (default overlay dir `analysis/ortholog_policy/v1` when present).
6. The new run must still pass CLI hash checks — the UI does not unlock Block by itself.

CLI-equivalent after Block (for comparison): [tokenization.md § Ortholog loss gate](tokenization.md#ortholog-loss-gate).

### Ortholog loss gate — approval card (Web UI)

When cross-species tokenize hits `ortholog_loss_gate` **Block**, Pipeline exits non-zero and the Analysis **Outputs** section shows an **approval required** card **only if** `ortholog_approval_request.yaml` is on disk under the current run / last run folder.


| Principle            | Detail                                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------------------ |
| CLI is authoritative | Hash checks / Block / `validate_approval_record` live in `core/geneformer/ortholog_loss_gate.py`       |
| UI is a recorder     | Never clears Block by itself; writes `approval_record.yaml` then (optionally) starts a **new** run     |
| Pending ≠ approved   | Do not reuse `ortholog_approval_request.yaml` as an approval path — the gate rejects `status: pending` |


**What the card shows:** direction, critical gene, `present_in_input`, mapping status, presented candidates, excluded paralog (e.g. POU5F1B), and a short impact line — from the pending request, not from free text.

**How to use it (step by step):**

1. Confirm the table matches the gene you care about (and that it was present in input — otherwise the gate would only Warn).
2. Enter **Approved by** (operator name; no login) and **Reason** (both required).
3. Pick one decision, then **Record decision**:


| Choice                                                | Writes                                                                                                                   | Re-run?                                                                                                       |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| **Stop / keep strict one2one**                        | `approval_record.yaml` with `decision: stop`, `status: approved`                                                         | No                                                                                                            |
| **Approve curated bridge (presented candidate only)** | Same file with `decision: curated_bridge`; pins overlay `sha256`; uses **only** candidates already listed on the request | **Yes** — new Pipeline with `ortholog_approval_record` + `ortholog_curated_overlay` (+ audit path when found) |
| **Reject candidate**                                  | `status: rejected` + reason                                                                                              | No                                                                                                            |


1. Optional **Dismiss card** hides the card for this request path in the current session (does not delete artifacts).

`best_of_n` is **not** a one-click UI action. For a sensitivity run, start a separate Pipeline and set ortholog policy to **best_of_n** in Species / model.

**Artifacts after Block / decision** (under the tokenize output dir for that run):


| File                                                     | Role                                       |
| -------------------------------------------------------- | ------------------------------------------ |
| `ortholog_approval_request.yaml`                         | Pending ticket from the blocked run        |
| `approval_record.yaml`                                   | UI/CLI decision (`approved` or `rejected`) |
| `ortholog_loss_summary.json` / `critical_gene_audit.tsv` | Verdict detail                             |


Session id is stored on the record as `ui_session_id`. Implementation: `[webui/ortholog_approval_ui.py](../webui/ortholog_approval_ui.py)`. Gate rules: [tokenization.md § Ortholog loss gate](tokenization.md#ortholog-loss-gate).

### Analysis / Output tabs

Under the page title:


| Tab          | Purpose                                                                                                                                                                        |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Analysis** | Data input, run type, YAML, **Run job**, live log, and the current-session Outputs section                                                                                     |
| **Output**   | Browse past `pipeline_`* folders under `/app/output` **only** (dropped-gene table, figures inline, CSV previews, zip download). Works even if the Analysis session / Streamlit connection was lost |

**Method description.** Under **Run directory** on Analysis, and again for the selected past run on Output, a paragraph is ready to copy into Methods. It states the ISP Platform version, whether the job is the end-to-end pipeline, the input species, the model, ISP mode (`delete` or `overexpress`), Start and Goal, and the fine-tune batch. After tokenization it adds the median, mean, SD and range of token length (genes kept per cell, truncated at the model input size). ISP UMAP and State-feedback ISP fill those fields from the selected pipeline run and add the UMAP genes or the ordered steps. Edits in the box are not saved.

### One job at a time, and stopping a job

Only one job runs at a time across **all** browser sessions. **Run job** takes a lock file, `{WEBUI_WORKSPACE}/active_job.json`, and is disabled in every session while the job runs. A tab opened later, or the same tab after a reload, shows the running job under **Logs & status** (run type, run folder, log tail).

**Stop job → Stop now** (next to **Refresh log**) stops the job and its child processes: SIGTERM to the job's process group, then SIGKILL after 15 s. Files already written stay in the run folder, but the run is incomplete; the log shows `stopped by signal 15`.

The lock clears itself when the job's process exits, including after a Web UI or container restart. It covers Web UI jobs only: a CLI job (`docker compose run …`) on the same GPU is not blocked.


---



## Guides per run type


| Run type                  | Doc                                                                                                          |
| ------------------------- | ------------------------------------------------------------------------------------------------------------ |
| FT batch size (calibrate) | Measures GPU; recommends Pipeline `runtime.train_batch_size` ([fine-tuning.md](fine-tuning.md) § batch size) |
| Pipeline (E2E)            | [pipeline.md](pipeline.md)                                                                                   |
| ISP UMAP                  | [isp_umap.md](isp_umap.md) — pick a **past Pipeline ISP run** + gene (not Data input zip)                    |
| State-feedback ISP        | [state_feedback_isp.md](state_feedback_isp.md) — pick a **past Pipeline ISP run** + ordered OE/KD steps; mode **Placebo-corrected (paper method)** (default) or **Decoder diagnostics** |

What State-feedback ISP does, and every setting (conditions, feedback, specificity), is explained in [state_feedback_isp.md](state_feedback_isp.md) (v1.0.1).


**Study Data input** stays loaded when switching between **FT batch size (calibrate)** and **Pipeline (E2E)**. **ISP UMAP** and **State-feedback ISP** hide Data input and instead list completed `pipeline_*/stage_configs/isp.yaml` runs under `/app/output`.

### Fine-tune `train_batch_size`

1. Run type **FT batch size (calibrate)** → **Run job** (~1 min, no training). **GPU must be idle** (stop any other Pipeline / fine-tune first).
2. Copy the recommended integer (or on Pipeline click **Use calibrated (N)**).
3. On **Pipeline (E2E)**, open **Advanced options** and set **Fine-tune train_batch_size**. Keep that value fixed for the study.

Changing FT batch size after you start comparing runs changes optimization dynamics; do not retune casually. ISP **ISP GPU batch size** (`forward_batch_size`) is independent.

### Fine-tune / ISP knobs on Pipeline (E2E)

Always visible: **ISP perturbation** (type / state_key / genes) and **Fine-tune task** (task_type / label_column).

Under **Advanced options** (collapsed by default):


| UI                                | Writes                                                                          |
| --------------------------------- | ------------------------------------------------------------------------------- |
| epochs / learning_rate / num_runs | `stages.finetune.training.*`                                                    |
| warmup                            | `training.warmup_steps: 500` (virtual genetic screen) or `warmup_ratio: 0.05` (classification) |
| tokenize max_cells                | `runtime.max_cells`                                                             |
| max_ncells                        | `stages.isp.isp.max_ncells` (default 2000; same idea as Mouse-Geneformer-WebUI) |
| stats.mode                        | `stages.isp.stats.mode` (usual: `goal_state_shift`)                             |
| ISP analysis plots                | `stages.isp.analysis.enabled`                                                   |


**Trajectory UMAP** (per-cell arrows): use Run type **ISP UMAP**. Choose a past E2E pipeline folder + gene; toggle **Draw trajectory lines** (and arrow count) under Plot options. Optional expander **Cluster / cell-type analysis** enables `cluster_coexpr_analysis/` after the UMAP (`n_clusters` can be a fixed K or `auto`). The job runs `run_isp_umap.py --run-dir … --gene …` and writes under `{pipeline_run}/isp_umap/`.

**State-feedback ISP** (OE then KD, or any ordered list of steps): use Run type **State-feedback ISP**. Choose the same past pipeline folder and add steps (`overexpress` / `delete` + gene lists); after each step, a Δrank decoder reorders each cell's genes from the model output before the next step. The default mode, **Placebo-corrected (paper method)**, runs `core/run_state_feedback_stability.py` with the paper's two-tier placebo correction (10 correction placebos, 30 comparison placebo chains, decoder seeds 0–2), reports τ per seed and records the placebo genes (`estimation_placebos.json`, `random_chains.json`, also shown and downloadable in Outputs). For one step, **Cluster / cell-type / enrichment analysis** (the same switch as Pipeline (E2E)) adds enrichment of the top 10% genes of the start cells, the goal-state cells and ISP (Δspec), computed locally on Reactome, GO BP, MF and CC, and KEGG, and closeness to Goal (how far ISP moves Start toward Goal per gene and per gene set, against the placebo chains). See [state_feedback_isp.md](state_feedback_isp.md) 3.4. **Decoder diagnostics (no placebo correction)** runs `core/run_state_feedback_isp.py` for direction fidelity and the six conditions; its endpoint values are not τ. Writes under `{pipeline_run}/state_feedback_isp/`. Read [state_feedback_isp.md](state_feedback_isp.md) before choosing conditions; design and results are in [state_feedback_decode_methods.md](state_feedback_decode_methods.md).

Pipeline (E2E) additionally runs **TOP1 significant ISP UMAP** automatically once (same output directory; gene = largest shift toward goal among FDR-significant genes detected in ≥20 cells, as in the significant-genes lollipop). Full trajectory UMAP remains available via Run type **ISP UMAP**.

Also on the main form (not under Advanced):


| UI                                               | Writes                                                                                               |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| perturbation.type / state_key / genes_to_perturb | `perturbation.*`                                                                                     |
| task_type / label_column                         | `stages.finetune.finetune.*` (`label_column` dropdown from sample-folder metadata + tokenizer attrs) |
| Cluster / cell-type / enrichment analysis | Above **ISP model**, not inside Advanced. `stages.isp.postprocess.enabled` (default **off**), `n_clusters` (default **auto**), `celltype_prediction`, `rank_enrichment`. Rank enrichment of the top Start, Goal and ISP genes goes to `rank_enrichment/` and is shown in **Output browser** ([pipeline.md](pipeline.md)) |
| Fine-tune batch | Above **ISP GPU batch size**. Auto (`ft_batch_choice: auto`) keeps effective batch size 6 and, when one pass of 6 cells does not fit, uses micro-batch (4096 tokens: 2 cells per pass, 3 passes). Manual keeps the standard 4–8 batch separate from micro-batch. A **Methods** box copies the sentence |


These merge onto `config/finetune.yaml` / `config/isp.yaml` at pipeline start. Changing epochs/LR/num_runs/warmup, `max_ncells` / `stats.mode`, perturbation type/genes, or FT labels changes results; analysis only toggles optional post-stats figures. Empty `genes_to_perturb` = all genes (slow); a short list (e.g. `Igfbp2`) enables targeted ISP (and optional auto UMAP if YAML `umap.enabled` is true). Default warmup is **500 steps** (virtual genetic screen). Choose **rate 0.05** in Advanced options when the fine-tune goal is classification.

Tokenize, fine-tune, and standalone ISP: use CLI (`docker compose run --rm tokenize` / `finetune` / `isp`) — see [tokenization.md](tokenization.md), [fine-tuning.md](fine-tuning.md), [in-silico pertabation.md](in-silico%20pertabation.md).

Ad-hoc one-off Python (rare; build profile):

```bash
docker compose --profile build run --rm --no-deps isp-platform python3 /app/scripts/...
```

