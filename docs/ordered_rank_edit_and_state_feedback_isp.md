# Ordered rank-edit ISP and State-feedback ISP — how they work and how to use them

*Added in v1.0.1.*

The platform has two ways to apply several perturbations in order.

| | Ordered rank-edit ISP | State-feedback ISP |
|---|---|---|
| What carries into the next step | The **token edits** made so far (edited gene ranks) only | The model output after the perturbation, **decoded back into a gene order** |
| Model output reused | No (each step's shift is only read out) | Yes (feedback) |
| Training | None | A Δrank decoder is trained per run (or reused) |
| Role | Default path used in the paper | Extension (evaluated in model space only) |

This document explains, chapter by chapter, **what each method does and what it does not do**. Both are readouts in the representation space of a fine-tuned model. **Neither simulates how a cell changes over time.**

Details and measured results are in [ordered_rank_edit_isp.md](ordered_rank_edit_isp.md) (Ordered rank-edit configuration reference) and [state_feedback_decode_methods.md](state_feedback_decode_methods.md) (State-feedback design and validation).

---

## 1. Shared background

### 1.1 Rank-value encoding

Geneformer reads a cell as a list of gene tokens sorted from the highest to the lowest expression rank (`input_ids`). The leftmost token has the highest rank. Expression values are not given to the model; **only the order** carries information.

### 1.2 Perturbations (OE / KD)

In both methods, one step is a group perturbation of one or more genes applied together.

- **overexpress (OE):** moves the genes to the front of the list. Length is preserved: as many tokens as were inserted are cut from the end. A gene already in the list is removed from its old position and moved to the front.
- **delete (KD):** removes the genes from the list, so the list gets shorter.

### 1.3 goal_state_shift (scoring)

The score measures how much closer the perturbed cell embedding moves to the centroid of the goal state (the pipeline's end state).

```text
shift = cos(perturbed cell embedding, goal centroid) − cos(original cell embedding, goal centroid)
```

- The comparison is always against the **original start-cell encoding**, not the previous step.
- A positive value means the cell moved toward the goal state.

### 1.4 Fine-tuning is required

ISP results are only reliable with a model fine-tuned for cell-state classification. In the Web UI both methods start from a finished **Pipeline (E2E)** run and reuse its tokenized dataset, fine-tuned model and start / end states.

---

## 2. Ordered rank-edit ISP

### 2.1 What it does

It applies OE / KD steps to start-state cells one at a time, in the order you give. After each step it scores and writes the goal_state_shift of the encoding at that point.

```text
start encoding X0
  step 1: X1 = edit(X0, genes of step 1)    -> write shift(X1 vs X0)
  step 2: X2 = edit(X1, genes of step 2)    -> write shift(X2 vs X0)
  ...
  step T: XT = edit(XT-1, genes of step T)  -> write shift(XT vs X0)  (final shift)
```

`edit` is only the token operation from 1.2.

### 2.2 Intermediate states are reported, but the model's response is not passed to the next step

The shift after each step (the 1st-step result, the 2nd-step result, …) is written out, but **it is a readout; it is never used as input to the next step.**

- The input to step 2, X1, is the original encoding with the step-1 token edit applied. How the model responded at step 1 (hidden states or the shift value) is not reflected in X1 at all.
- So the method does **not** work as "step 1 changed the cell like this, so apply step 2 to that changed cell". The step-1 intermediate state is simply **the encoding with the token edits up to step 1, passed through the model**.
- A curve of step results shows how the readout changes as edits are added one by one. It does not represent a time course or a stepwise change in cell state.

To feed the model's response into the next step, use State-feedback ISP (chapter 3).

### 2.3 The final shift depends only on the final encoding — which genes were edited, and in what order

The final shift is determined by the final encoding XT alone. Whether intermediate steps were scored, and what their shifts were, has no effect on it.

- **The same final encoding gives the same final shift.** Building XT directly without scoring the intermediate steps gives the same result.
- **Order matters only when it changes the final encoding.**
  - OE only: the later a gene is overexpressed, the further left it sits. A then B gives `[B, A, …]` at the front; B then A gives `[A, B, …]`. The final encodings differ, so the shifts can differ. The differences between the 24 OSKM orders come from this **arrangement at the front of the encoding**.
  - KD only: deletion does not depend on order, so the final encoding is the same and order has no effect on the final shift.
  - OE and KD on the same gene: OE then KD removes the gene; KD then OE puts it back at the front. Here too, order acts only through the final encoding.
- A simultaneous group OE (one step with `[A, B]`) gives `[A, B, …]` (the first gene in the list is leftmost). This is the reverse of applying A then B in order, which gives `[B, A, …]`.

**The "order effect" is therefore the effect of different gene arrangements near the front of the final encoding on the fine-tuned model's readout.** Do not interpret it as the biological order of factor delivery or as temporal causation.

### 2.4 Notes on scoring

- For step lists that are OE only or KD only, scoring uses the standard Geneformer group ISP score (the perturbed genes are aligned with the original encoding before comparing).
- When OE and KD are mixed the positions cannot be aligned, so scoring uses the cell-mean cosine (mean embedding over all tokens) without alignment.
- Each step runs two forward passes (perturbed and original encoding), so it uses more GPU memory than genome-wide ISP. `batch=auto` uses a separate measurement (see the `batch=auto` section of [ordered_rank_edit_isp.md](ordered_rank_edit_isp.md)).

### 2.5 Web UI settings

Choose the Run type **Ordered rank-edit ISP**.

| Setting | Meaning |
|---|---|
| ISP source run (left column) | Pipeline (E2E) run to reuse. It sets the dataset, fine-tuned model and start / end states |
| Number of steps | Number of steps |
| Step N: type | `overexpress (OE)` or `delete (KD)` |
| Step N: name | Tag added to the output folder name (optional) |
| Step N: genes | Genes perturbed together in this step (one per line; symbols or Ensembl IDs) |
| max_ncells | Number of start-state cells used |
| Save intermediate perturbed datasets | Save the perturbed dataset after each step (needed only for UMAP; off by default) |
| GPU batch size | `auto` (recommended) or manual. Reusing the genome-wide ISP batch size often causes out-of-memory errors |

**Apply run + steps to Config YAML** writes the settings into the YAML; **Run job** runs it.

### 2.6 Outputs

Written under `{pipeline_run}/ordered_rank_edit_isp/…/`.

- `steps/stepNN_<name>/single_gene_per_cell_shifts.csv` — per-cell shift after each step (`Shift_to_goal_end`)
- `step_summary.csv` — median / mean / quartiles / fraction positive per step
- `run_manifest.json` — settings and run information

---

## 3. State-feedback ISP

### 3.1 What it does

It applies the same kind of step list as Ordered rank-edit, but after a chosen step it **turns the model's response back into a gene order and uses that as the input to the next step**.

```text
start encoding X0
  step 1: X1 = edit(X0, step 1)                              -> write shift
  feedback: pass X0 and X1 through the model; take the per-gene hidden-state change Δh
            Δh -> decoder -> per-gene rank displacement -> reorder the genes of X1 into X1'
                                                             -> write shift
  step 2: X2 = edit(X1', step 2)                             -> write shift
  ...
```

- Only **the order of the genes already in each cell** is changed. No genes are added or removed.
- The reordered X1' is what the next perturbation acts on. This is the essential difference from Ordered rank-edit.
- By default feedback is applied after `Feedback after step` and after every later step, including the last, so each perturbation acts on the order the model predicted after the previous one and the endpoint is the reordered encoding (3.7). Genes overexpressed so far stay at the front; only the other genes are reordered. Switching multi-step off gives one feedback event; the model output is still fed into the next input, so it is still State-feedback ISP.

### 3.2 What "decode" means

Geneformer has no head that outputs expression. For each gene token the model returns a hidden state (a vector of several hundred dimensions), which is not a gene order. To pass the model's response to the next step, the hidden states must be **translated into the input format (a gene order)**. That translation is the decode step.

- The decoder outputs **not expression values but how far, and in which direction, to move each gene (Δrank)**.
- Its input is the perturbation-induced **change** in hidden state, Δh = h(perturbed) − h(reference), not the reference hidden state itself. This keeps the decoder reading the perturbation effect rather than the cell's baseline identity.
- It cannot be said that expression is recovered from hidden states. What can be said is that "from the perturbation-induced change in representation, the decoder predicts the **direction** of the observed rank change beyond what the original rank alone explains" (3.6).

### 3.3 Training the Δrank decoder

The decoder is trained per run from that run's dataset and model (an existing decoder can also be reused; see 3.7).

**Teacher signal (what counts as correct)**

1. For the start-state cells and the observed-state cells (the goal state by default), compute each gene's **mean position** across cells (pseudobulk). Normalize it to 0–1 by each group's mean encoding length.
2. The per-gene observed rank displacement `Δr_obs = normalized position(observed) − normalized position(start)` is the teacher. Positive means the gene moved right (lower rank); negative means it moved left (higher rank).
3. Genes detected in fewer than `min_detection_count` cells (default 5) in either group are dropped, because their mean position is unstable.

This is **a difference between the start and observed-state groups**, not a measurement of cells that received the same perturbation. Per-perturbation observed ranks (for example from time-course data) are not available in this dataset and are not used.

**Training samples**

1. Pass the encoding with **all configured steps applied at once**, and the original encoding, through the fine-tuned model (up to `train_max_ncells` cells, default 200).
2. For each gene present in both, take Δh (genes are matched by token, not position) and pair it with its teacher value `Δr_obs` (up to 256 genes per cell). The gene's original normalized position `base_rank` is kept for the base-rank control but is not a decoder input.
3. Split the **genes** (not the cells) 80:20. The 20% are never used for training and are kept for evaluation. Because the teacher is one value per gene, splitting by cell would leak the answers.

**Model**

```text
Δrank    = max_shift · tanh( w·Δh )     # one linear layer on Δh only, no bias, bounded
priority = base_rank + Δrank            # lower = further left
new order = genes sorted by ascending priority
```

- The decoder reads Δh only (no base rank, no bias term). Zero perturbation (Δh = 0) therefore gives exactly zero displacement for any trained weights, and a gene's movement cannot depend on its position alone.
- `max_shift` caps how far one feedback event can move a gene, as a fraction of the cell's encoding length. It limits the decoder's output, not Geneformer: `0.5` means one event can move a gene up to half of the cell's gene list.
- Weights are initialized to zero, so an untrained decoder leaves the order unchanged.
- Decoders saved before this change also read base rank and a bias. They cannot be loaded; retrain instead of passing them to `--decoder-checkpoint`.

**Loss**

- Huber loss (predicted Δrank vs Δr_obs)
- Ordering loss (whether the Δrank order of gene pairs matches the teacher)
- Smoothness penalty (discourages extreme displacements)

**Choosing `max_shift`**

A decoder is trained for each value in `max_shift_grid` (default 0.05 / 0.1 / 0.2 / 0.3 / 0.5). The one with the highest Spearman correlation between predicted Δrank and Δr_obs on the held-out 20% of genes is kept. Results go to `decoder/decoder_metrics.json` and the weights to `decoder/delta_rank_decoder.pt`.

### 3.4 One feedback event, step by step

1. Pass the reference encoding (the original start encoding by default) and the encoding after the last step through the model.
2. Take Δh per gene and compute a score with the condition's method (3.5).
3. Reorder the cell's genes by ascending `priority = base_rank + bounded displacement`.
4. Score the reordered encoding and use it as the input to the next step.

Notes:

- The genes just overexpressed are also reordered. The displacement is capped by `max_shift`, so a gene placed at the front cannot fall far back, but it can move a little from the front.
- Within a State-feedback run, all conditions are scored with the **cell-mean cosine without alignment**, because reordering does not keep the perturbed genes at fixed positions. So the `ordered_rank_edit` condition inside a State-feedback run does not match the numbers from running Ordered rank-edit ISP on its own (which uses the group ISP score). **Compare only within the same run.**

### 3.5 Conditions — what each one does and why it is there

One run compares the selected conditions on **the same cells and the same step list**. By default all six are run. Only `linear_deltarank` is the method; the others are comparisons, a ceiling and a safety check. **Do not report the result of any condition other than `linear_deltarank` as the State-feedback ISP result.**

| Condition | What the feedback uses | Role | How to read it |
|---|---|---|---|
| `ordered_rank_edit` | Nothing (no feedback) | Baseline: the same step list as Ordered rank-edit | Reference for whether a method moves cells closer to the goal |
| `norm` | Change in each gene's hidden-state length (L2 norm) | Training-free comparison | If the decoder cannot clearly beat it, the decoder's value is not shown |
| `delta_mlm` | Change in the pretrained MLM head's logit for each gene token | Training-free comparison | Same as above |
| `linear_deltarank` | The trained Δrank decoder (3.3) | **The method** | Judged by direction fidelity (3.6) |
| `oracle` | Reorders each cell's genes by the observed state's mean positions | Rough ceiling | Not a performance target (see below) |
| `null_feedback` | The decoder applied with no perturbation (reference and input are both the original encoding) | Safety check | Normal if the order does not change at all and the shift is 0 |

**`norm` (hidden-state norm)**

- For each gene it computes `‖h(perturbed)‖ − ‖h(reference)‖`, z-scores it within the cell, and moves genes whose norm grew further left. The displacement is capped at `baseline_max_shift` (default 0.1).
- It tests the common but unfounded reading "a larger embedding means a more important or more highly expressed gene", using the same reordering machinery as the decoder. Only the scalar being read differs, so the comparison is fair.
- Hidden-state norm does not represent expression. On BBRC, its correlation with the observed rank change of held-out genes was close to zero (ρ ≈ 0.03).

**`delta_mlm` (MLM self-logit difference)**

- Using the MLM head of the pretrained (not fine-tuned) model, it computes, before and after the perturbation, the logit for "this gene token belongs at this position", and moves genes whose logit rose further left. The displacement cap is the same as for `norm`.
- The MLM head is not updated during fine-tuning, so it is mismatched with the fine-tuned representations. Taking a difference may cancel part of the mismatch, but what it measures is a change in token plausibility, not a change in expression rank.
- On BBRC its correlation was indistinguishable from chance (ρ ≈ −0.02), yet its endpoint shift was the largest. This is the clearest example of why **a method must not be chosen by endpoint shift alone**.

**`linear_deltarank` (the method)**

- The decoder trained in 3.3 predicts each gene's displacement from its Δh and original rank, and reorders the genes.
- It is judged mainly by direction fidelity (3.6); endpoint shift is secondary.

**`oracle` (ceiling)**

- Keeps each cell's gene set and reorders it by the mean positions observed in the observed-state cells. It uses neither the model nor the decoder.
- It puts goal-state information directly into the input, so moving toward the goal is nearly guaranteed (endpoint leakage). It indicates "how far reordering can move the model space"; **do not use it as a target for the decoder or as a result of the method**.
- It is the denominator of `gap_closed_fraction` in the endpoint gate.

**`null_feedback` (safety check)**

- Runs the decoder with no perturbation (reference and input are the same original encoding). Δh = 0, so a correct decoder changes nothing.
- If the order changes or the shift moves away from 0, the decoder has learned an unfounded reordering. On BBRC the measured displacement was 0 and the shift was 0.

### 3.6 Evaluation — how the method is judged

**Primary axis: direction fidelity (shown as PASS / not passed in the Web UI Outputs)**

On the 20% of genes not used for training, it compares each method's predicted rank change with the observed rank change Δr_obs by rank correlation.

- PASS requires both of the following.
  1. `linear_deltarank` beats both `norm` and `delta_mlm`, and the 95% confidence interval of each difference (cell-level bootstrap) excludes 0.
  2. The correlation after removing what the original rank explains (partial ρ given base rank) has a 95% confidence interval above 0.
- Condition 2 is needed because the teacher is a group-mean rank difference, so much of the correlation can be explained by position alone ("genes that start high tend to go down, genes that start low tend to go up"). The partial ρ is the information Δh adds beyond the original position.
- Measured on BBRC (n=300): pooled ρ 0.437, partial ρ 0.242 [0.234, 0.250], PASS.

**Secondary: endpoint gate (`phase12_gate.csv`)**

For each condition it shows how much of the gap between `ordered_rank_edit` (baseline) and `oracle` (ceiling) the median final-step shift closes (`gap_closed_fraction`). Endpoint shift cannot tell "reordered in the right direction" from "the encoding was disturbed", so **do not judge a method by this alone** (see the `delta_mlm` example in 3.5).

**Optional: perturbation specificity**

The same decoder is given Δh from other gene sets and from random gene perturbations, and the partial ρ values are compared. If the configured perturbation's partial ρ is above the distribution for random perturbations, the gain depends on the perturbation's Δh rather than on properties of the scored genes.

### 3.7 Web UI settings

Choose the Run type **State-feedback ISP**. ISP source run, steps and GPU batch size are the same as for Ordered rank-edit (2.5).

**Basic settings**

| Setting | Default | Meaning |
|---|---|---|
| max_ncells | 300 | Number of start-state cells. Every condition runs GPU forward passes over these cells, so run time is roughly proportional (observed: about 26 min in total and about 4 min per condition at n=50) |
| Δrank decoder | Train a new decoder in this run | Train a new decoder, or reuse one trained earlier on the same Pipeline run (same fine-tuned model). The steps it was trained on are shown in brackets. **Reusing it for a different step list is a transfer test; performance is not guaranteed.** This includes the same genes in a different order: the gene added last ends up at the front of the encoding, so the Δh the decoder sees changes. On BBRC OSKM (n=3000, all 24 orders, feedback after every step), a decoder trained on simultaneous OSKM matched a decoder trained on each order only for the 6 orders ending in POU5F1, whose encodings match the simultaneous one at the front. On the other 18 its direction fidelity (pooled Spearman) fell to 0.32–0.36, against 0.42–0.47 for per-order decoders (mean over 24 orders: 0.38 vs 0.46; partial ρ given base rank 0.29 vs 0.35). The endpoint order ranking from the two decoders was uncorrelated (Spearman −0.02) |
| Direction fidelity only | Off | When on, the endpoint conditions are skipped and only decoder training (or loading) and direction fidelity (and specificity) run. Works with a single step |

**Conditions and feedback**

| Setting | Default | Meaning |
|---|---|---|
| Conditions | All six | Conditions to run (3.5). The `linear_deltarank` verdict needs `norm` and `delta_mlm`, and the gap needs `ordered_rank_edit` and `oracle`, so **keeping all of them selected is recommended** |
| Observed state for the teacher | Blank = pipeline end state | State that supplies the decoder's teacher and the `oracle` ranks. It must exist in the same dataset under the same state key, with enough cells. If the dataset has an intermediate-state label, it can be used here |

**Feedback after every step**

Feedback is applied after every step, including the last. There is no other schedule and no cap, so the number of feedback events always equals the number of steps. Example with 4 steps:

KLF4 → **feedback** → MYC → **feedback** → SOX2 → **feedback** → POU5F1 → **feedback**

This is what makes State-feedback ISP sequential: each step acts on the state the model inferred after the previous step. Without feedback between steps the result depends only on the final encoding. That is Ordered rank-edit ISP, which equals multi-gene ISP with the gene list reversed. Feedback only after the last step also depends only on the final encoding. The endpoint conditions therefore need at least 2 steps.

> **Caution: the error grows with the number of steps.** The decoder is trained to predict the whole observed start→end rank change from the Δh with all steps applied. Each feedback event (with `ctrl_reference: start`) recomputes Δh from the start encoding, including earlier steps and earlier reorders, and adds another end-point-scale displacement to an order that has already moved. Nothing undoes it. Displacement and decoder error therefore grow with the number of steps, and past the observed end-state ranks the shift reflects encoding disturbance rather than the biology the model has learned. Part of the gain is not specific to the chosen genes: on BBRC OSKM (4 steps), random genes gained most of what OSKM gained (docs/state_feedback_decode_methods.md, "Multi-step stability evaluation"). **Keep chains short, compare only chains with the same number of steps, and before a biological claim subtract matched random chains** (`core/run_state_feedback_stability.py`). This random effect is large and needs to be reduced in future versions.

There is no per-cell stop. A convergence stop (Spearman > 0.995 twice in a row) was removed: each step adds a new perturbation, so small feedback changes so far do not mean the next step's feedback will be small, and a whole-encoding threshold misses large moves of a few genes (in a 2048-gene cell, one gene moving from the bottom to the top still gives Spearman 0.997). A 2-cycle stop was also removed: it required the whole order to return exactly to the one from two events ago, which practically never happens because a new perturbation enters between events.

**Perturbation specificity (optional)**

| Setting | Meaning |
|---|---|
| Feed the same decoder Δh from other perturbations | Run the specificity evaluation (3.6) |
| Named sets | Gene sets to compare (one per line: `name: GENE1 GENE2 …`) |
| Named-set type | Apply the sets as OE or KD |
| Random draws / Genes per random draw / Random type | Number of random perturbations, genes per draw, and their type |
| Match random genes' detection rate to | Restrict random genes to those detected in 0.5–2× as many start cells as this gene. Recommended for KD, since deleting an undetected gene changes nothing |

### 3.8 Settings available only in the YAML

These can be changed by editing the **Config YAML** in the Web UI. The defaults are normally fine.

| Key (under `state_feedback.`) | Default | Meaning |
|---|---|---|
| `pin_overexpressed` | `true` | After each reorder, genes overexpressed so far stay at the front (in their order before the reorder); only the other genes move |
| `ctrl_reference` | `start` | Reference for Δh. `start` is the original start encoding (the cumulative effect so far); `previous` is the encoding before the last step (the last step's effect only). Decoder training corresponds to `start` |
| `hysteresis` | 0.0 | Genes whose displacements differ by less than this keep their original relative order (prevents jitter swaps). 0 disables it |
| `alpha`, `baseline_max_shift` | 1.0, 0.1 | Scale and cap of the displacement for `norm` / `delta_mlm` (no effect on the decoder) |
| `observed_max_ncells`, `min_detection_count` | 3000, 5 | Maximum cells used to build the teacher, and minimum number of cells a gene must be detected in to be used |
| `decoder.*` | See 3.3 | Training cells, genes per cell, epochs, loss weights, `max_shift_grid`, held-out gene fraction |
| `eval.*` | | Direction-fidelity settings (top-K, bootstrap and permutation counts) |
| `stability.*` | | Random-chain control and stability evaluation (`core/run_state_feedback_stability.py`): decoder seeds, `n_random_chains`, `random_seeds` (one independent draw each), `min_stratum` |

`feedback_every_step`, `feedback_after_step`, `feedback_after_last_step` and `multi_step` were removed. A config that still sets them to anything other than feedback after every step stops with an error.

### 3.9 Outputs

Written under `{pipeline_run}/state_feedback_isp/state_feedback_isp_<time>/`.

| Path | Contents |
|---|---|
| `decoder/` | The trained decoder (`delta_rank_decoder.pt`) and results per `max_shift` (`decoder_metrics.json`) |
| `direction_fidelity/` | Primary evaluation: `direction_fidelity.json` (verdict), per-method correlations and precision@K, base-rank control, confidence intervals of method differences |
| `<condition>/stepNN_<name>/per_cell_shifts.csv` | Per-cell shift for each condition and step |
| `<condition>/stepNN_feedback/per_cell_shifts.csv` | Shift right after feedback |
| `phase12_summary.csv` | Shift summaries and reordering diagnostics for every condition and step (including feedback) |
| `phase12_gate.csv` | Endpoint gate (secondary) |
| `perturbation_specificity/` | Only when specificity is on |
| `run_manifest.json` | Settings, decoder information, gate, multi-step counts |

The Web UI Outputs panel shows the direction-fidelity verdict, the per-method table, the specificity tables, the endpoint gate and the multi-step counts.

### 3.10 Running with the defaults, and what to check

The defaults (all six conditions, `Feedback after step` = 1, multi-step on including the last step, max_ncells 300, train a new decoder) include every condition needed for the comparison.

1. Look at the **direction-fidelity verdict** first. If it does not pass, the decoder cannot be said to read the direction of rank change, however large the endpoint shift is.
2. If it passes, use the endpoint gate as secondary information. The `oracle` value is a rough ceiling, not a target.
3. Check that `null_feedback` has a shift of 0 and an unchanged order.
4. To save time, reuse a decoder trained on the same Pipeline run with the same step list (for example to add conditions or specificity). Using it on a different step list than it was trained on is a transfer test; train a new decoder for results you report (see the Δrank decoder row in 3.7).

---

## 4. Which one to use

| Goal | Method |
|---|---|
| Get the readout for genes perturbed in order, as in the paper | Ordered rank-edit ISP |
| See how perturbation order (the arrangement at the front of the final encoding) changes the readout | Ordered rank-edit ISP |
| Feed the model's response to a perturbation into the input of the next perturbation | State-feedback ISP |
| Test whether the perturbation-induced change in representation predicts the direction of observed rank change | State-feedback ISP (Direction fidelity only is enough) |

---

## 5. What can and cannot be claimed

**Can be claimed**

- Ordered rank-edit ISP: how much an encoding, with the given genes edited in the given order, moves toward the goal state in the fine-tuned model's space.
- State-feedback ISP: a decoder trained on perturbation-induced hidden-state changes predicts the direction of observed rank change beyond what the original rank explains (held-out genes; BBRC and Asano PIPseq). Also, the readout when the encoding reordered by that prediction is used for the next perturbation.

**Cannot be claimed**

- That either method simulates a cell's time course or stepwise state change. The intermediate-step values of Ordered rank-edit are not the result of passing the model's response forward.
- That the order effect in Ordered rank-edit reflects the biological order of factor delivery.
- That hidden-state norm or MLM logits represent expression, or that the decoder recovers expression.
- That the `oracle` value is a result or a target of the method.
- That multi-step feedback is more useful or more stable than a single feedback event (not evaluated).
- That the **size** of the decoder's displacements is accurate (the direction matches, but the size is compressed).
