# Pretrained model weights

Large binaries are **not** committed (see `.gitignore` → `/models/`).

## Build then run (recommended)

`docker compose build geneformer-platform` downloads weights + dictionaries into the image
(`/opt/geneformer-assets`). On container start, `scripts/container-entrypoint.sh` seeds
them into `./models` and `core/geneformer/dicts/` when those host paths are empty/incomplete.
Host copies always win (local override).

```bash
docker compose build geneformer-platform
docker compose up -d platform_webui
```

| `DOWNLOAD_MODELS` | What gets baked |
|-------------------|-----------------|
| `default` (compose default) | Mouse base + 12L-E20 + Human V2-104M + mouse/human dicts + mouse↔human orthologs |
| `minimal` | Mouse base + mouse dicts |
| `all` | `default` + fly ortholog BioMart tables |
| `none` | Skip (use manual scripts below) |

Optional: `HF_TOKEN` build-arg / `.env` if Hugging Face rate-limits anonymous downloads.
Build failure prints a clear error (network / Drive / HF / BioMart). Plan ~1 GB free for a default bake.

Orchestrator: [`scripts/download_build_assets.sh`](../scripts/download_build_assets.sh).

## Manual download (host or `DOWNLOAD_MODELS=none`)

### Mouse Geneformer

Native species: **mouse**. Config: `species.model: mouse_geneformer`.

| Variant (`species.mouse_variant`) | Web UI | Directory | Size / architecture |
|-----------------------------------|--------|-----------|---------------------|
| `base` (default) | Base (6L / ~10M) | `mouse-Geneformer/` | ~10M params; 6L / 256 / 2048 / SiLU |
| `12l_e20` | Large (12L-E20) | `mouse-Geneformer-12L-E20/` | 12L / 256 / 2048 / SiLU (same vocab as base) |

```bash
bash scripts/download_mouse_geneformer.sh           # base + 12L + token/median/symbol dicts (+ safetensors)
bash scripts/download_mouse_geneformer.sh --12l     # 12L only
bash scripts/download_mouse_geneformer.sh --base    # base only
bash scripts/download_mouse_geneformer.sh --prune-bin  # drop pytorch_model.bin after safetensors
```

Upstream Google Drive (same archives the script fetches):

- [mouse-Geneformer (base)](https://drive.google.com/file/d/1gM3gcc3DlNGt5bAcqHbeRxtdMktGeDEg/view?usp=sharing)
- [mouse-Geneformer-12L-E20](https://drive.google.com/file/d/1xKMyFA4JJeRigcJPsU2XNyxEW25Q247u/view?usp=sharing)

Mouse dictionaries land under `core/geneformer/dicts/mouse/` (token + median + symbol→Ensembl). See that folder’s README.

### Human Geneformer

Native species: **human**. Config: `species.model: human_geneformer`.

| Variant (`species.human_variant`) | Web UI | Directory | Size / architecture | Bake |
|-----------------------------------|--------|-----------|---------------------|------|
| `v2_104m` (default) | Base (V2-104M) | `human-Geneformer-V2-104M/` | ~104M params; 12L / 768 / 4096 | `default`, `all` |
| `v2_316m` | Large (V2-316M) | `human-Geneformer-V2-316M/` | ~316M params; 18L / 1152 / 4096 | not baked — HF manual |

```bash
bash scripts/download_human_geneformer_v2_104m.sh
```

There is **no Drosophila pretrained checkpoint** — fly scRNA-seq uses ortholog remapping into mouse or human models. **Drosophila / fly input is Beta** (Web UI: **Drosophila (fruit fly) (Beta)**): not biologically validated; experimental only.

### Orthologs and dictionaries (supporting assets)

Not pretrained models — gene vocab / cross-species maps used at tokenize and ISP:

```bash
bash scripts/download_mouse_human_orthologs.sh
bash scripts/download_drosophila_orthologs.sh   # optional; DOWNLOAD_MODELS=all
```

Tiny `*_curated.tsv` overrides and `fly_symbol_to_fbgn.tsv` are tracked in git. Platform overview of model choices: [README — Available models](../README.md#available-models).

## Example species blocks

```yaml
# Mouse data → mouse 12L
species:
  model_organism: mouse
  model: mouse_geneformer
  mouse_variant: 12l_e20

# Human data → mouse 12L (ortholog conversion)
species:
  model_organism: human
  model: mouse_geneformer
  mouse_variant: 12l_e20

# Mouse data → Human V2
species:
  model_organism: mouse
  model: human_geneformer
  human_variant: v2_104m
```
