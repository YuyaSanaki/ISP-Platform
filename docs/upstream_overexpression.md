# Group overexpression: ISP Platform vs official Geneformer

ISP Platform overexpresses a group of genes with a **length-preserving** rule
(`overexpress_tokens` in `core/geneformer/in_silico_perturber.py`). This note records how that rule
differs from the official Geneformer implementation, and the one input range where the official
implementation misaligns the cells it compares.

## What was tested

| Item | Value |
| --- | --- |
| Code | Hugging Face `ctheodoris/Geneformer`, commit `1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5` (the tip of `main` when tested on 2026-09-26 and still on 2026-09-30) |
| Entry point | Stock `InSilicoPerturber`: `perturb_type="overexpress"`, `genes_to_perturb` = POU5F1, SOX2, KLF4, MYC (Ensembl IDs), `combos=0`, `emb_mode="cell"`, `cell_emb_style="mean_pool"`, no goal states |
| Model | Geneformer V2-104M fine-tuned as a CellClassifier on GSE147564 (somatic vs pluripotent), input size 4,096 |
| Cells | 16 per case; lengths set per case (see below) |
| Host | NVIDIA H100 |

## Result

| Case | Cell length | OSKM in the cell | Official Geneformer |
| --- | --- | --- | --- |
| 1 | 4,096 | all absent | runs |
| 2 | 4,094 | all absent | `RuntimeError: The size of tensor a (4092) must match the size of tensor b (4090) at non-singleton dimension 1` |
| 3 | 4,096 | all present | runs |
| 4 | 4,096 | 2 present, 2 absent | runs |
| 5 | < 800 | all absent | runs |

The error is raised in `perturber_utils.quant_cos_sims` when the gene embeddings of the perturbed and
original cells are compared.

## Why case 2 fails

Official Geneformer inserts the genes at the front and cuts the perturbed cell to the model input
size `max_len` (`perturber_utils.overexpress_tokens`). To keep the two cells comparable, it then cuts
the original cell by `n_overflow` (`calc_n_overflow`, `truncate_by_n_overflow`, the block the
reviewer linked, `in_silico_perturber.py` L605). `n_overflow` is computed from the length **after**
the cut, not before. With k OE genes absent from a cell of length L:

- L = `max_len`: the count is right (k).
- L ≤ `max_len` − 2k: the count is ≤ 0 and nothing is cut.
- `max_len` − 2k < L < `max_len`: the count is too large, the original loses genes it should keep,
  and the two cells differ in length. For OSKM with all four genes absent this is L = 4,089–4,095.
  Case 2 (L = 4,094): perturbed 4,096 − 4 = 4,092 genes, original 4,094 − 4 = 4,090 genes.

The band was obtained by applying the official functions to every length from `max_len` − 20 to
`max_len` (k = 1–4, with and without present OE genes). From the code: without goal states (and with
`emb_mode="cell_and_gene"`) the gene embeddings are compared and the run stops, as in case 2. With goal
states and `emb_mode="cell"` the cell embeddings are averaged separately, so the run completes, but
the cells in the band are compared with an original that lacks its last genes.

The band is narrow. In the GSE147564 dataset used in the paper (Human-Geneformer V2-104M
tokenization), 0 of 3,000 somatic cells (median length 790) and 1 of 3,000 pluripotent cells have a
length of 4,089–4,095.

## How ISP Platform differs

| | Official Geneformer (`1f7fbae`) | ISP Platform |
| --- | --- | --- |
| Length of the perturbed cell | grows by the inserted genes, then is cut to `max_len` | equals the cell's own length before OE; net insertions drop the lowest-ranked genes |
| Alignment for comparison | cut the original by `n_overflow` | remove exactly as many unique positions from the original as genes were overexpressed (`oe_indices_to_remove_for_alignment`) |
| Depends on the cell length | yes (band above) | no |

The ISP Platform perturber was derived from the Mouse-Geneformer code, which inserted OE genes without
cutting and aligned absent genes by removing the last positions of the batch. When a present OE
gene already sat in those last positions, fewer positions were removed for that cell and stacking
the batch failed with a shape mismatch (e.g. [2044, 256] vs [2045, 256]). The length-preserving rule
and the unique-position alignment replace that path.

Tests: `tests/test_overexpress_length_preserve.py` (`TestUpstreamOverflowBand` runs the case 2 input
and the whole band through the ISP Platform operators).
