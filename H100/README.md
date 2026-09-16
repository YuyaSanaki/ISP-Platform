# Running ISP Platform on NVIDIA H100 (x86_64)

Standalone guide for **Geneformer tokenize → fine-tune → ISP** on an **H100 (~80 GB)** machine: HPC cluster, cloud GPU instance, or bare metal.

**Runtime:** native **conda +** `python3`.  **Not** using Docker, Apptainer, or Singularity.

Site-specific details (MFA, queue names, filesystem layout) stay in your center’s docs. Scripts in this folder are scheduler-agnostic cores plus Slurm / PBS examples.

---

## What you need

- Linux **x86_64** with **1× NVIDIA H100** and working drivers (`nvidia-smi`)
- Disk for repo + models + data + outputs (plan **≥50 GB** free to start)
- User-writable **conda** (Miniconda / Mambaforge / site module)
- Clone access to [ISP-Platform](https://github.com/YuyaSanaki/ISP-Platform)
- scRNA-seq in 10x-style folders: `barcodes.tsv.gz`, `features.tsv.gz`, `matrix.mtx.gz`

Optional on HPC: batch scheduler (**Slurm** or **PBS**), `module load cuda/...`.

Observed Human Geneformer ISP peak VRAM: about **35–41 GiB** (fits one H100 80GB).

---

## Files in this folder


| File | Role |
| --- | --- |
| [`env.sh`](env.sh) | Shared `ISP_ROOT` / conda activate |
| [`setup_env.sh`](setup_env.sh) | Create conda env `isp` + PyTorch + deps |
| [`download_models.sh`](download_models.sh) | Download weights, dicts, ortholog tables |
| [`smoke_cuda.sh`](smoke_cuda.sh) | Check GPU + `torch.cuda` |
| [`run_pipeline.sh`](run_pipeline.sh) | Run `core/run_pipeline.py` |
| [`job_slurm.sh`](job_slurm.sh) | Example Slurm job |
| [`job_pbs.sh`](job_pbs.sh) | Example PBS-style job |

Scripts are marked executable in git (`100755`). Prefer invoking with **`bash`** (portable on HPC); `./H100/….sh` also works after clone:

```bash
bash "$H100_DIR/setup_env.sh"
# or: "$H100_DIR/setup_env.sh"
```

Suggested layouts on the H100 host (either works; `env.sh` auto-detects both):

**A — toolkit inside the clone (recommended):**

```text
$WORK/ISP-Platform/     # git clone
  H100/                 # this toolkit
  core/
  ...
```

```bash
export ISP_ROOT="$WORK/ISP-Platform"
export H100_DIR="$ISP_ROOT/H100"
```

**B — toolkit as a sibling copy (optional):**

```text
$WORK/
  ISP-Platform/     # git clone
  H100/             # copy of this toolkit
  logs/
```

```bash
export ISP_ROOT="$WORK/ISP-Platform"
export H100_DIR="$WORK/H100"
```

`$WORK` is any writable project directory (`$HOME/scratch`, `/work/$USER`, `/workspace`, …).

---

## Step-by-step

### Step 0 — Checklist

- [ ] On a **GPU node**, `nvidia-smi` shows an H100 (not only a CPU login node, if your site separates them)
- [ ] You can write under `$WORK`
- [ ] Outbound network for `git`, `pip`, and model downloads (or you have offline copies)

### Step 1 — Shell on a GPU node

**Cloud / dedicated box:** SSH in; that machine is the GPU node.

**HPC:** allocate a GPU interactively, then continue:

```bash
# Slurm (typical)
srun --gres=gpu:1 --time=01:00:00 --pty bash

# PBS / center-specific interactive (names vary)
# qsub -I ...
# qlogin ...
```

Avoid long `pip` installs and full ISP runs on shared login nodes if policy forbids it.

### Step 2 — Clone the code and place this toolkit

```bash
mkdir -p "$WORK" && cd "$WORK"
git clone https://github.com/YuyaSanaki/ISP-Platform.git

# Copy this H100/ directory next to the clone if it is not already on the machine
# (e.g. rsync from your laptop, or include it in your project tree)
```

```bash
export WORK=...                                    # your choice
export ISP_ROOT="$WORK/ISP-Platform"
export H100_DIR="$WORK/H100"
```

### Step 3 — Create the Python environment (once)

```bash
# Optional on HPC:
# export CUDA_MODULE=cuda/12.3.2

bash "$H100_DIR/setup_env.sh"
bash "$H100_DIR/smoke_cuda.sh"
```

This creates conda env `**isp**` (override with `CONDA_ENV=...`), installs CUDA-enabled **PyTorch**, then the rest of ISP Platform dependencies (pinned `torch` / `nvidia-`* lines in `requirements.txt` are skipped so they do not fight the CUDA wheel).

Expect `smoke_cuda.sh` to print `cuda_available True` and an H100 device name.

### Step 4 — Download models and dictionaries

A fresh clone does **not** include large weights. Download them onto the host:

```bash
export ISP_ROOT="$WORK/ISP-Platform"
# optional: export DOWNLOAD_MODELS=default   # mouse + Human V2-104M + orthologs
# optional: export HF_TOKEN=hf_xxx
bash "$H100_DIR/download_models.sh"
ls "$ISP_ROOT/models/"
```

### Step 5 — Add data and write a config with **host paths**

1. Put your study under `$ISP_ROOT/data/<study>/` (one folder per sample with the three 10x files).
2. Create a pipeline YAML with **absolute paths on this machine**.
  Example configs under `core/config/` are templates; copy one and set paths to `$ISP_ROOT`:

```bash
cp "$ISP_ROOT/core/config/pipeline_1w_human_v2.yaml" \
   "$ISP_ROOT/core/config/my_study.yaml"
# then edit my_study.yaml
```

```yaml
data:
  input_dir: "/absolute/path/to/ISP-Platform/data/my_study/"
  output_prefix: "my_study"

paths:
  output_root: /absolute/path/to/ISP-Platform/output/my_study

species:
  model_organism: mouse          # species of the input matrix
  model: human_geneformer        # or mouse_geneformer
  human_variant: v2_104m         # when using human_geneformer
  ortholog_policy: one2one       # when input species ≠ model species

perturbation:
  type: delete
  state_key: disease
  start_state: AD                # match your metadata
  end_state: WT
  genes_to_perturb: []   # [] = genome-wide (very long) or specify gene like "Inr"
```

Use real absolute paths (expand `$ISP_ROOT` yourself). Relative `/app/...` strings in the repo templates are leftovers from other deployments — **they will fail here**; always point at your checkout.

### Step 6 — Interactive test run

```bash
export ISP_ROOT="$WORK/ISP-Platform"
export PIPELINE_CONFIG="$ISP_ROOT/core/config/my_study.yaml"
bash "$H100_DIR/run_pipeline.sh"
```

Or CUDA-only: `bash "$H100_DIR/smoke_cuda.sh"`.

### Step 7 — Batch job (production)

Edit `ISP_ROOT`, `PIPELINE_CONFIG`, account / partition / walltime in the job script, then submit.

**Slurm:**

```bash
sbatch "$H100_DIR/job_slurm.sh"
squeue -u "$USER"
```

**PBS-style:**

```bash
qsub "$H100_DIR/job_pbs.sh"
qstat
```

Both call `run_pipeline.sh` after activating conda. Keep large logs under `$WORK/logs/`.

### Step 8 — Outputs

```bash
ls "$ISP_ROOT/output/"
# copy home if needed
rsync -avz "$ISP_ROOT/output/"  your-laptop:~/isp-outputs/
```

---

## Environment variables


| Variable          | Meaning                                      |
| ----------------- | -------------------------------------------- |
| `ISP_ROOT`        | Absolute path to ISP-Platform (**required**) |
| `CONDA_ENV`       | Default `isp`                                |
| `PIPELINE_CONFIG` | YAML for `run_pipeline.py`                   |
| `DOWNLOAD_MODELS` | Default `default` for model download         |
| `HF_TOKEN`        | Optional Hugging Face token                  |
| `CUDA_MODULE`     | Optional `module load` name                  |
| `WANDB_DISABLED`  | Default `true`                               |
| `TORCH_INDEX_URL` | PyTorch wheel index (default cu121)          |


---

## Troubleshooting


| Symptom                                                | What to check                                                      |
| ------------------------------------------------------ | ------------------------------------------------------------------ |
| `torch.cuda.is_available()` is False                   | GPU allocation? Driver? `CUDA_MODULE`? Correct conda env `isp`?    |
| OOM / killed on login node                             | Use a compute / GPU allocation for install and runs                |
| `No such file` under data/output                       | YAML still has wrong paths — use absolute `$ISP_ROOT/...` (Step 5) |
| Job hits walltime                                      | Increase scheduler time; genome-wide ISP is long                   |
| Broken `torch` after `pip install -r requirements.txt` | Use `setup_env.sh` (it filters conflicting pins)                   |


---

## Done when

- [ ] `nvidia-smi` shows H100 on the node you run on
- [ ] `smoke_cuda.sh` → `True` + H100 name
- [ ] Target checkpoint exists under `models/`
- [ ] Your YAML uses absolute host paths
- [ ] `run_pipeline.sh` (or a batch job) writes under `ISP-Platform/output/`
