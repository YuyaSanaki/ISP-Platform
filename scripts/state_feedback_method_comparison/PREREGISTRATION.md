# Pre-specified criteria: method comparison at n=300 (x7c_methods_n300)

Copied into the repository on 2026-10-11 from the working directory of the analysis. The text is
unchanged except that cluster job commands were replaced by the equivalent `run_methods.sh` call.
File names in backticks that are not in this folder (`REMEASURE_BUDGET.md`, `subsample_x7c_methods.py`,
`x7c_n50`, `random_path_n300`) refer to that working directory. "Fig. 1" in Addendum 3 is Fig. 4 of
the revised manuscript.

Fixed 2026-10-01 14:36 JST, before any stage-2 (loop) result of this run existed: at
14:35:30 there were 0 `loop` lines in the logs and 0 `stage2_per_cell_*.csv` files. Stage-1
(decoder-level) results for seeds 0/1 had been looked at; they are not used by the criteria below.

Run: `run_methods.sh` with n=300 and 30 evaluation placebos, 3 chains per job (submitted 14:10),
`x7_causal.py`, code `9cc9f74`, n=300 cells, decoder seeds 0/1/2, 10 estimation placebos (draw
seed 1), 30 evaluation placebos (draw seed 0), positive controls OSK and OSNL, loop variants
ORE, V0, V1, V2, med_generic, med_specific, OSKM decoder for every chain. Metrics as defined in
`analyze_x7c_methods.py`.

Role: confirmation of the method chosen on the n=50 screen, and the method comparison for the
revision. It uses the screening script; the numbers reported for the platform will be rerun with
the implementation in `core/` and must reproduce these.

## Primary method: potential outcome + Delta-spec (loop V2, outcome Delta-spec)

Confirmed only if all hold:

- C1: OSKM ranks 1 among OSKM + 30 placebos in each of the 3 seeds.
- C2: separation z >= 3 in each seed.
- C3: positive controls OSK and OSNL have z > 2 against the placebos in each seed.
- C4: OSKM beats every placebo in >= 90% of cells (mean over seeds).

## Comparison claims

- C5 (vs Delta-spec only = V0 + Delta-spec): z of the primary method is higher in each seed.
- C6 (vs SCM backdoor = V1, and vs SCM mediation = med_specific, each with and without
  Delta-spec): z of the primary method is higher in each seed. If it is not higher than SCM
  mediation in every seed, the two are reported as comparable, not as the primary being better.
- C7 (why Delta-spec is added to the potential-outcome loop): for potential outcome only (V2 raw),
  the mean placebo estimate is > 30% of the OSKM estimate in each seed.

## Descriptive, no threshold

- Mediation: med_generic / V0 gain for OSKM and for placebos is reported as measured; the n=50
  value (0.99-1.00) is restated only if the n=300 value supports it.
- Decoder-level direction fidelity is reported next to the placebo-trained decoders; it is not a
  criterion.

## Rules

- No change to n_est, ridge, the placebo draws, the variant definitions or the metrics after
  results are seen. Anything added later (e.g. n_est = 5 / 20 sensitivity) is labelled post hoc.
- Failed criteria are reported as failed.
- Jobs that crash are rerun for the missing chains only (`--tag ...r`, resume), with the same
  settings.

## Addendum 1 (2026-10-01 14:45 JST): is n=100 enough for the 24-order experiment (X6)?

Added before any primary-method result existed. At 14:45 the logs held 10 `loop` lines, all
loop V0 (raw chain gains for OSKM in two seeds and 8 evaluation placebos), and there were no
V1 / V2 / med loop lines and no `stage2_per_cell_*.csv`. The criteria C1-C7 above are unchanged.

Background: X6 (24 orders x 30 order-matched placebos x 3 seeds) does not fit the remaining
budget at n=300; n=100 was chosen for X6 on budget grounds. The n=50 variance split put about
99% of the variance of the estimate in the placebo-chain spread, so z should barely depend on n.
This check tests that on n=300 data.

Start cells are taken as the longest cells first (`_select_start_cells` sorts by length), so an
X6 run with 100 cells uses cells 0-99 of this run. Cells 0-99 are therefore a fixed subset
here, in addition to random subsets.

Procedure, after the run completes, per seed, for the primary method (V2 + Delta-spec):

- Subsets: cells 0-99, and B = 1000 random 100-cell subsets drawn without replacement
  (`numpy.random.default_rng(0)`).
- On each subset, recompute C1, C2, C4 (and C3 descriptively) with the same metric code,
  restricting OSKM, placebos and positive controls to the same cells. Decoders stay as fitted
  on 300 cells, so this checks the outcome side only, not decoder training at n=100.
- Report per seed: the pass rate of each criterion over the random subsets, the result on cells
  0-99, z and the 95% CI half-width of the OSKM estimate at n=100 vs n=300 (CI as in
  `specific_gain`: cells and placebo chains resampled).

Decision rule: n=100 is kept for X6 if, in each seed, C1, C2 and C4 hold on cells 0-99 and in
>= 95% of the random subsets. Otherwise the X6 design is revised before submission (more cells
with fewer placebos, or fewer orders), and this result is reported either way.

Not part of this addendum: the power for order-to-order differences is estimated from the
existing `random_path_n300` data (already analysed, V0 decoder); it is a descriptive estimate,
not a pre-specified test.

## Addendum 2 (2026-10-01 15:37 JST, cluster clock): X6 with 20 placebos per order

Reason: measured V2 cost (888 s per chain at n=300, vs ~12 min assumed) puts X6 at n=100 with 30
placebos at ~225 node-hours, which does not fit; with 20 placebos it is ~153
(`../REMEASURE_BUDGET.md`). What had been seen when this was added: chain-level loop lines in
the logs (raw gains, no Delta-spec, no per-cell data) for 14 V2 runs, including OSKM in seed 0
(raw gain 0.36). Two `stage2_per_cell_*.csv` files (seed 0 g10, 15:35:24; seed 1 g1, 15:36:35)
had been written by the jobs but were not opened; no Delta-spec or criterion had been computed.

Addition to Addendum 1: run the same check with 20 placebos as well
(`subsample_x7c_methods.py --n-placebo 20`): cells 0-99 with placebos eval0-eval19, and for each
of the B random cell subsets an independent random draw of 20 of the 30 placebos (Delta-spec
recomputed within the draw). The decision rule of Addendum 1 applies to this 20-placebo check;
the 30-placebo check is reported next to it. C1-C7 are unchanged.

## Addendum 3 (2026-10-08, post hoc): LEACE loop variants V1b and V1+V1b at n=300

Post hoc. Added after all results of this run (C1-C7, the method table) had been seen, so that
the method comparison in Fig. 1 covers every loop correction screened at n=50 (`x7c_n50`):
V1b (LEACE erasure of base rank) and V1+V1b (V1 then LEACE) were screened there but left out
of this run. Nothing is refitted: the loop jobs reuse each seed's stage1 `state.pt` of this run
(the V1b and V1+V1b decoders and LEACE maps were already fitted and saved by stage1), with the
same cells, chains, chain groups, placebo draws and code (`9cc9f74`).
Run: loop jobs with `--variants V1b,V1+V1b`, tags `b0`-`b10`.
Reported descriptively next to the other methods, with the same metrics; C1-C7 are unchanged
and are not re-evaluated for these variants.
