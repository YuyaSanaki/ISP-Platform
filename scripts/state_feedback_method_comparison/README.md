# State-feedback ISP: comparison of placebo corrections (Fig. 4A, B)

Code for the comparison of correction methods in Fig. 4A, B of the revised manuscript: OSKM added
one factor per step in the order KLF4 → MYC → SOX2 → POU5F1 to 300 somatic cells (GSE147564,
Geneformer V2-104M fine-tuned on somatic cells and P20P iPSCs), decoder seeds 0, 1 and 2, compared
with 30 evaluation placebo chains matched for detection and rank in the start cells.

The method used in the paper, potential outcome + Δspec, is implemented in `core/`
(`core/run_state_feedback_stability.py` with `state_feedback.placebo_contrast.enabled: true`) and is
the one to use for new analyses. This folder holds the analysis code that compared it with the
alternatives. It is not part of the platform and is not maintained.

| File | Role |
| --- | --- |
| `x7_causal.py` | Runs the comparison. `--mode stage1`: decoders and correction maps per seed. `--mode loop`: feedback after every step for each chain and correction |
| `analyze_x7c_methods.py` | Method table (`method_metrics.csv`): estimate with 95% cell-bootstrap CI, placebo mean / estimate, z, rank, cells above every placebo |
| `run_methods.sh` | The calls used for the paper (n = 300, 30 evaluation placebos, 10 estimation placebos) |
| `PREREGISTRATION.md` | Criteria fixed before the results were seen |

## Corrections (names in the code → Fig. 4B)

Notation: X = hidden-state change of a gene caused by the perturbation (perturbed minus start
encoding), Z = hidden state of the gene in the start encoding.

| Loop variant | Decoder input | Fig. 4B |
| --- | --- | --- |
| `V0` | X | No correction; with the Δspec outcome: Δspec only |
| `V1` | X − f(Z), f a ridge regression fitted on the estimation placebos | Backdoor adjustment |
| `med_specific` | `V0` decoder fed X − m, m = mean X of the estimation placebos swapped in for the factors in the current encoding | Mediator adjustment |
| `V2` | decoder trained and fed X − m (potential-outcome contrast) | PO only; with the Δspec outcome: PO + Δspec (the platform method) |
| `ORE` | no feedback | Baseline of the feedback gain |

Outcome: raw, the mean over cells of the feedback gain (end point with feedback minus the same steps
without feedback), or Δspec, the per-cell gain minus the mean gain of the placebo chains
(leave-one-out for a placebo).

## Reproducing

The comparison ran with the code of commit `9cc9f74` (an ancestor of tag `v1.0.1`), which still has
the no-feedback runner `core/run_ordered_rank_edit_isp.py` that `x7_causal.py` imports.

```bash
git worktree add ../code_9cc9f74 9cc9f74
bash scripts/state_feedback_method_comparison/run_methods.sh \
  "$(realpath ../code_9cc9f74)" <tokenized_dataset> <fine_tuned_model> <out>
python3 scripts/state_feedback_method_comparison/analyze_x7c_methods.py <out>
```

Each loop call takes about 15 min per chain and correction on one H100 GPU at n = 300.
