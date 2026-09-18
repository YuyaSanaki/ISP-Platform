# Fine-tuning Geneformer for classification

This guide describes the **end-to-end workflow** to fine-tune Mouse-Geneformer on **cell type classification** or **disease classification** tasks. The fine-tuned model can then be used by the ISP service for more targeted perturbation analysis.

Entrypoint: [`run_finetune.py`](../core/run_finetune.py). Configuration: [`core/config/finetune.yaml`](../core/config/finetune.yaml). Docker: [`docker-compose.yml`](../docker-compose.yml).

---

## 1. Why fine-tune?

The pretrained Mouse-Geneformer has **general-purpose** cell embeddings learned from 20M mouse cells. Fine-tuning adapts these embeddings for a **specific downstream task**:

| Use case | What fine-tuning does |
|----------|----------------------|
| **Disease classification** (Ctrl vs Disease) | Embeddings learn to separate disease states — ISP then identifies genes whose perturbation shifts cells between states |
| **Cell type classification** | Embeddings learn cell-type boundaries — ISP then identifies cell-type-defining genes |

### Pipeline flow

```
tokenize → finetune → isp (with fine-tuned model)
```

Without fine-tuning, ISP uses the pretrained model (`model.type: Pretrained`). After fine-tuning, ISP uses the checkpoint (`model.type: CellClassifier`) — perturbation scores become more specific to your experimental question.

---

## 2. Prerequisites

Before fine-tuning you need:

1. **A tokenized `.dataset`** — produced by the tokenize service (see [tokenization.md](tokenization.md)). The dataset must have:

   | Column | Required? | Notes |
   |--------|-----------|-------|
   | `input_ids` | **Yes** | Rank-encoded gene tokens per cell |
   | `length` | **Yes** | Sequence length |
   | Label column | **Yes** | E.g. `disease` (with values like `Ctrl`, `Disease`) or `cell_type` |
   | Organ/group column | Optional | E.g. `organ_major` — used for per-organ fine-tuning |

2. **The pretrained model** — default: `/app/models/mouse-Geneformer/` (contains `config.json`, `pytorch_model.bin`).

3. **Docker image built** — `docker compose build isp-platform`.

### What if my dataset is missing columns?

The fine-tuning config has a **metadata injection** section that can add, rename, or derive columns at runtime (before training). You do **not** need to re-tokenize.

---

## 3. Configure `config/finetune.yaml`

The config has six sections. Here is each with explanation:

### 3.1 Paths

```yaml
paths:
  dataset: /app/data/MyStudy/tokenized_dataset/MyStudy_0.dataset
  geneformer_model: /app/models/mouse-Geneformer/
  output_root: /app/output
  output_time_subdir: true
```

| Key | Description |
|-----|-------------|
| `dataset` | Path to the tokenized HuggingFace `.dataset` directory inside the container |
| `geneformer_model` | Pretrained model directory to fine-tune from |
| `output_root` | Parent output directory; runs are written to `{output_root}/{YYYYMMDD}/finetune_<UTC>/` |
| `output_time_subdir` | `true` (default): create a timestamped subfolder per run to avoid overwrites |

### 3.2 Metadata injection

Use this when your dataset is missing columns needed for fine-tuning. Applied **before** training.

```yaml
metadata:
  # Rename existing columns
  rename_columns: {}
    # Example: { genotype: cell_type }

  # Add a fixed-value column to every row
  add_columns:
    organ_major: brain

  # Derive a column from another using a value mapping
  # derive_columns:
  #   cell_type:
  #     source: genotype
  #     mapping:
  #       Ctrl: control
  #       Disease: disease_model
```

| Operation | When to use | Example |
|-----------|-------------|---------|
| `rename_columns` | Column exists but has the wrong name | `{ genotype: cell_type }` — renames `genotype` to `cell_type` |
| `add_columns` | Column is completely missing | `{ organ_major: brain }` — adds `organ_major = "brain"` to all rows |
| `derive_columns` | Need a new column computed from an existing one | Derive `cell_type` from `genotype` with a value mapping |

### 3.3 Fine-tuning task

```yaml
finetune:
  task_type: disease           # "disease" or "cell_type"
  label_column: disease        # column to use as classification labels
  organ_key: null              # group by this column, or null for full dataset
  rare_class_threshold: 0.005  # drop classes < 0.5% of total cells
  train_split: 0.8             # 80/20 train/eval split
  shuffle_seed: 42
```

| Key | Description |
|-----|-------------|
| `task_type` | `disease` or `cell_type` — determines the classification objective |
| `label_column` | Which dataset column becomes the label. Must exist (or be created via `metadata`) |
| `organ_key` | If set (e.g. `organ_major`), fine-tuning runs **per organ group**. Set to `null` to train on the full dataset as one group |
| `rare_class_threshold` | Classes representing less than this fraction are dropped (prevents training instability) |
| `train_split` | Fraction of data used for training (rest is eval) |

**Example configurations:**

| Scenario | `task_type` | `label_column` | `organ_key` |
|----------|-------------|----------------|-------------|
| Disease classification, no organ grouping | `disease` | `disease` | `null` |
| Disease classification, per organ | `disease` | `disease` | `organ_major` |
| Cell type classification, per organ | `cell_type` | `cell_type` | `organ_major` |
| Cell type classification, whole dataset | `cell_type` | `cell_type` | `null` |

### 3.4 Model settings

```yaml
model:
  max_input_size: 2048
  freeze_layers: 0
  ignore_mismatched_sizes: true
```

| Key | Description |
|-----|-------------|
| `max_input_size` | Maximum token sequence length (2048 = 2^11, matches Geneformer default) |
| `freeze_layers` | Number of bottom BERT encoder layers to freeze (0 = train all layers). Freezing layers speeds training and can prevent overfitting on small datasets |
| `ignore_mismatched_sizes` | Allow loading pretrained weights when the classifier head size differs (required when the pretrained model has a different number of output classes) |

**Layer freezing guide:**

| Dataset size | Recommended `freeze_layers` |
|--------------|----------------------------|
| < 5,000 cells | 4–6 (freeze most, avoid overfitting) |
| 5,000–50,000 cells | 2–4 |
| > 50,000 cells | 0 (train everything) |

### 3.5 Training hyperparameters

```yaml
training:
  learning_rate: 5e-5
  batch_size: 6
  eval_batch_size: 2
  lr_scheduler_type: linear   # linear | cosine | polynomial
  warmup_ratio: null          # set 0.05 for classification-oriented FT (overrides warmup_steps)
  warmup_steps: 500           # default: virtual genetic screen / ISP
  epochs: 10
  weight_decay: 0.001
  fp16: true
  seed: 42
  num_runs: 1
```

| Key | Description |
|-----|-------------|
| `learning_rate` | Peak learning rate (5e-5 is a good starting point for BERT fine-tuning) |
| `batch_size` | Per-device training batch size. Prefer a value from Web UI **FT batch size (calibrate)** / `run_ft_batch_calibrate.py`, then **keep it fixed** — changing it alters optimizer step counts and results |
| `eval_batch_size` | Per-device eval batch size (can be larger than train batch) |
| `lr_scheduler_type` | `linear` (standard), `cosine` (smooth decay), or `polynomial` |
| `warmup_steps` | Steps of linear warmup before decay. **Default 500** (v1.1.0) — use for virtual genetic screens / ISP gene-effect size |
| `warmup_ratio` | If set (e.g. `0.05`), overrides `warmup_steps`. Use **rate 0.05** when the fine-tune goal is classification, not per-gene ISP |
| `epochs` | Training epochs (10 is typical; reduce for large datasets) |
| `weight_decay` | L2 regularization strength |
| `fp16` | Mixed-precision training (faster on NVIDIA GPUs) |
| `seed` | Random seed for reproducibility |
| `num_runs` | Run training multiple times with incrementing seeds (for statistical robustness) |

> **Tip:** We highly recommend tuning hyperparameters for your specific dataset. See [`hyperparam_optimiz_for_cell_type_classifier.py`](../hyperparam_optimiz_for_cell_type_classifier.py) for a Ray Tune example.
>
> **Batch size stability:** After you pick an appropriate `training.batch_size` / Pipeline `runtime.train_batch_size` (Web UI **FT batch size (calibrate)**), keep that integer fixed for comparable fine-tunes. Do not flip between Auto and larger batches across runs you intend to compare.

### 3.5.1 Calibrate train_batch_size (no training)

```bash
python3 /app/core/run_ft_batch_calibrate.py --config /app/core/config/ft_batch_calibrate.yaml
# or from Web UI: Run type → FT batch size (calibrate) → Run job
```

Writes `recommended_train_batch_size.json` and prints `RECOMMENDED_TRAIN_BATCH_SIZE=<n>`. Paste `<n>` into Pipeline `runtime.train_batch_size`.

### 3.6 UMAP visualization

```yaml
umap:
  enabled: true
  n_neighbors: 15
  pca_components: 50
```

When enabled, generates a **PCA → UMAP** projection of the fine-tuned model's CLS embeddings on the eval set (last run only). Saved as PDF in the `figures/` subdirectory.

### 3.7 Runtime and provenance

```yaml
runtime:
  nproc: 16

provenance:
  enable_input_fingerprint: true
  fingerprint_fast: true
```

| Key | Description |
|-----|-------------|
| `nproc` | CPU workers for `datasets` map/filter operations |
| `enable_input_fingerprint` | Compute SHA-256 fingerprints of input data for reproducibility |
| `fingerprint_fast` | Use fast metadata-based fingerprinting (recommended for large datasets) |

---

## 4. Run fine-tuning

From the repository root:

```bash
docker compose run --rm finetune
```

This executes `python3 /app/core/run_finetune.py --config /app/core/config/finetune.yaml` inside the container with GPU access.

### Custom config

```bash
docker compose run --rm finetune python3 /app/core/run_finetune.py --config /app/core/config/my_custom_finetune.yaml
```

---

## 5. Outputs

All outputs are saved under `{output_root}/{YYYYMMDD}/finetune_<HHMMSS>_<microsecond>Z/`:

```
output/20260410/finetune_034500_123456Z/
├── finetune_config_used.yaml       # Copy of config used
├── finetune_run_metadata.yaml      # Timestamps, status, saved checkpoint paths
├── finetune_run.log                # Rotating stdout/stderr capture
│
├── all_run1/                       # Group name + run number
│   ├── config.json                 # Model config (for BertForSequenceClassification)
│   ├── model.safetensors           # Fine-tuned model weights
│   ├── training_args.bin           # HuggingFace TrainingArguments
│   ├── label_dict.json             # { "Ctrl": 0, "Disease": 1 }
│   ├── results.csv                 # True vs Predicted labels
│   ├── eval_results.json           # Accuracy, precision, recall, macro-F1
│   ├── preds.pkl                   # Full predictions object (pickle)
│   └── checkpoint-*/               # Epoch checkpoints (best model loaded at end)
│
├── best_all/                       # Best run results (when num_runs > 1)
│   └── best_results.csv
│
└── figures/
    └── all_umap.pdf                # PCA→UMAP projection plot
```

### Output metrics

The following metrics are computed and saved to `eval_results.json`:

| Metric | Description |
|--------|-------------|
| `accuracy` | Overall classification accuracy |
| `macro_precision` | Precision averaged across classes |
| `macro_recall` | Recall averaged across classes |
| `macro_f1` | F1 score averaged across classes (recommended for imbalanced datasets) |

---

## 6. Using the fine-tuned model with ISP

After fine-tuning, the script prints the checkpoint path. Update your ISP config to use it:

```yaml
# config/isp.yaml — before fine-tuning
paths:
  geneformer_model: /app/models/mouse-Geneformer/
model:
  type: Pretrained
  num_classes: 2
```

```yaml
# config/isp.yaml — after fine-tuning
paths:
  geneformer_model: /app/output/20260410/finetune_034500_123456Z/all_run1/
model:
  type: CellClassifier      # Changed from Pretrained
  num_classes: 2             # Must match the number of labels in fine-tuning
```

Then run ISP as before:

```bash
docker compose run --rm isp
```

The ISP service will now use the fine-tuned model's embeddings for perturbation analysis, producing results that are more specific to your classification task.

### label_dict.json

The fine-tuned model directory contains `label_dict.json` which maps class names to numeric IDs:

```json
{
  "Ctrl": 0,
  "Disease": 1
}
```

Ensure `perturbation.start_state` and `perturbation.end_state` in `isp.yaml` match the keys in this dictionary.

---

## 7. Per-organ fine-tuning

When `finetune.organ_key` is set (e.g. `organ_major`), the script loops over each unique value in that column and trains a separate model per group. This is useful when different tissues have different cell-type compositions.

Output structure with per-organ grouping:

```
output/20260410/finetune_.../
├── brain_run1/          # Model for brain
├── heart_run1/          # Model for heart
├── liver_run1/          # Model for liver
├── best_brain/
├── best_heart/
├── best_liver/
└── figures/
    ├── brain_umap.pdf
    ├── heart_umap.pdf
    └── liver_umap.pdf
```

---

## 8. Environment variables

| Variable | Meaning |
|----------|---------|
| `FINETUNE_CONFIG` | Override config path (default `/app/core/config/finetune.yaml`) |
| `WANDB_DISABLED` | Set to `true` (default in Compose) to disable Weights & Biases |
| `FINETUNE_LOG_MAX_BYTES` | Max size of `finetune_run.log` before rotation (default 50 MiB) |
| `FINETUNE_LOG_BACKUP_COUNT` | Rotated backups to keep (default `5`) |
| `FINETUNE_DISABLE_RUN_LOG` | Set to `1` / `true` / `yes` to skip file logging |
| `FINETUNE_GIT_COMMIT` | Optional commit SHA for metadata when `git` is unavailable |
| `FINETUNE_FINGERPRINT_INPUTS` | Set to `1` to compute input content fingerprints |

---

## 9. Troubleshooting

| Problem | Solution |
|---------|----------|
| `Label column 'X' not found in dataset` | Add the column via `metadata.add_columns` or `metadata.rename_columns` in config |
| CUDA OOM during training | Reduce `training.batch_size` (try 4, then 2) |
| CUDA OOM during UMAP | Set `umap.enabled: false` or reduce eval dataset size |
| Poor accuracy | Try: increase `training.epochs`, adjust `training.learning_rate` (1e-5 to 1e-4), try `lr_scheduler_type: cosine`, increase `training.warmup_steps` |
| Overfitting (train acc >> eval acc) | Increase `model.freeze_layers`, increase `training.weight_decay`, reduce `training.epochs` |
| Model checkpoint not usable by ISP | Ensure `model.type: CellClassifier` and `model.num_classes` matches `label_dict.json` |

---

## 10. Flow summary

**Tokenize → Configure `finetune.yaml` → `docker compose run --rm finetune` → Inspect metrics + UMAP → Update `isp.yaml` with checkpoint path → `docker compose run --rm isp` → Analysis.**
