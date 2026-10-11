#!/usr/bin/env python3
"""Compare ways of separating the OSKM-specific feedback effect from the placebo response.

  python3 analyze_x7c_methods.py <x7c_methods_dir>

Each method = a loop correction x an outcome-level contrast, applied to the per-cell gain over
Ordered rank-edit (feedback after every step, OSKM decoder):

  loop correction   none (V0) | SCM backdoor, context adjustment (V1) |
                    SCM mediator intervention, specific mediator only (med_specific) |
                    potential-outcome placebo contrast, contrast-trained decoder (V2)
  outcome           raw estimate E_c = mean_i gain_i(c)
                    Delta-spec   E_c = mean_i [gain_i(c) - mean_{p != c} gain_i(p)]  (placebos p,
                                 leave-one-out for a placebo c)

Named methods (main figure): Delta-spec only = V0 + Delta-spec; potential outcome only = V2 raw;
SCM backdoor = V1 raw; SCM mediation = med_specific raw; combination = V2 + Delta-spec.

Metrics per seed (placebo chains are the negative controls):
  oskm_estimate          E_OSKM [95% bootstrap CI over cells]
  placebo_mean_ratio     mean_p E_p / E_OSKM   (share of the reported effect a random gene also gets)
  placebo_max_ratio      max_p E_p / E_OSKM    (< 1: OSKM above every placebo)
  z                      (E_OSKM - mean_p E_p) / sd_p E_p
  rank, perm_p           rank of OSKM among OSKM + placebos, (1 + #p >= OSKM) / (n + 1)
  cell_top_frac          share of cells where OSKM's per-cell estimate beats every placebo
  pc_*_z                 positive controls (OSK, OSNL) against the placebos
Writes <dir>/analysis_methods/{method_metrics.csv, method_summary.csv, grid_summary.csv,
estimates.csv, mediation.csv, methods.png/pdf}.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

d = Path(sys.argv[1])
out = d / "analysis_methods"
out.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

files = sorted(d.glob("seed*/stage2_per_cell_*.csv"))
pc = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates(
    ["seed", "variant", "chain", "cell_index"], keep="last")
seeds = sorted(pc.seed.unique())

LOOPS = {"V0": "none", "V1": "SCM backdoor", "med_specific": "SCM mediation", "V2": "potential outcome"}
# Post hoc (PREREGISTRATION Addendum 3): LEACE loop variants, run later from the same state.pt.
POSTHOC = {"V1b": "LEACE base rank (post hoc)", "V1+V1b": "SCM backdoor + LEACE (post hoc)"}
if set(POSTHOC) & set(pc.variant):
    LOOPS = {"V0": LOOPS["V0"], "V1": LOOPS["V1"],
             **{v: lab for v, lab in POSTHOC.items() if v in set(pc.variant)},
             "med_specific": LOOPS["med_specific"], "V2": LOOPS["V2"]}
METHODS = [  # label, loop variant, outcome
    ("Delta-spec only", "V0", "dspec"),
    ("SCM backdoor", "V1", "raw"),
    ("SCM mediation", "med_specific", "raw"),
    ("Potential outcome only", "V2", "raw"),
    ("Potential outcome + Delta-spec", "V2", "dspec"),
]
GRID = [(f"{LOOPS[v]} | {o}", v, o) for v in LOOPS for o in ("raw", "dspec")]
rng = np.random.default_rng(0)


def per_cell_estimates(seed, variant, outcome):
    g = pc[(pc.seed == seed) & (pc.variant == variant)]
    w = g.pivot_table(index="cell_index", columns="chain", values="gain")
    plac = [c for c in w.columns if c.startswith("eval")]
    if outcome == "raw":
        return w, plac
    tot = w[plac].sum(axis=1)
    n = len(plac)
    est = pd.DataFrame(index=w.index)
    for c in w.columns:
        base = (tot - w[c]) / (n - 1) if c in plac else tot / n
        est[c] = w[c] - base
    return est, plac


def metrics(seed, label, variant, outcome):
    est, plac = per_cell_estimates(seed, variant, outcome)
    if "OSKM" not in est or len(plac) < 3:
        return None
    E = est.mean()
    eo, ep = E["OSKM"], E[plac]
    boots = [est["OSKM"].sample(frac=1, replace=True, random_state=int(s)).mean()
             for s in rng.integers(0, 1 << 31, 2000)]
    r = {"seed": seed, "method": label, "loop": variant, "outcome": outcome,
         "n_placebo": len(plac), "n_cells": len(est), "oskm_estimate": eo,
         "ci_lo": np.quantile(boots, 0.025), "ci_hi": np.quantile(boots, 0.975),
         "placebo_mean_ratio": ep.mean() / eo, "placebo_max_ratio": ep.max() / eo,
         "z": (eo - ep.mean()) / ep.std(), "rank": int(1 + (ep >= eo).sum()),
         "perm_p": (1 + (ep >= eo).sum()) / (len(plac) + 1),
         "cell_top_frac": float(est[plac].lt(est["OSKM"], axis=0).all(axis=1).mean())}
    for c in ("pc_OSK", "pc_OSNL"):
        if c in E:
            r[f"{c}_z"] = (E[c] - ep.mean()) / ep.std()
            r[f"{c}_ratio"] = E[c] / eo
    return r, E / eo, plac


rows, est_rows = [], []
for seed in seeds:
    for label, v, o in METHODS + GRID:
        m = metrics(seed, label, v, o)
        if m is None:
            continue
        r, rel, plac = m
        rows.append(r)
        if (label, v, o) in METHODS:
            est_rows += [{"seed": seed, "method": label, "chain": c,
                          "kind": "OSKM" if c == "OSKM" else ("placebo" if c in plac else c),
                          "relative_estimate": float(x)} for c, x in rel.items()]
allm = pd.DataFrame(rows).drop_duplicates(["seed", "method", "loop", "outcome"])
allm.to_csv(out / "method_metrics.csv", index=False)
ests = pd.DataFrame(est_rows)
ests.to_csv(out / "estimates.csv", index=False)

cols = ["oskm_estimate", "placebo_mean_ratio", "placebo_max_ratio", "z", "rank", "perm_p",
        "cell_top_frac", "pc_OSK_z", "pc_OSNL_z"]
cols = [c for c in cols if c in allm]
labels = [m[0] for m in METHODS]
main = allm[allm.method.isin(labels)].groupby("method")[cols].agg(["mean", "std"]).reindex(labels)
main.to_csv(out / "method_summary.csv")
grid = allm[allm.method.isin([g[0] for g in GRID])].groupby("method")[cols].mean().reindex([g[0] for g in GRID])
grid.to_csv(out / "grid_summary.csv")

med = pc[pc.variant.isin(["V0", "med_generic", "med_specific", "V2"])].groupby(
    ["seed", "variant", "chain"]).gain.mean().unstack("variant").reset_index()
med["kind"] = np.where(med.chain == "OSKM", "OSKM", np.where(med.chain.str.startswith("eval"), "placebo", med.chain))
if "med_generic" in med:
    med["generic_over_total"] = med["med_generic"] / med["V0"]
meds = med.groupby(["kind"]).mean(numeric_only=True)
meds.to_csv(out / "mediation.csv")

print(f"seeds {seeds}; cells {int(allm.n_cells.max())}; placebos per seed "
      f"{allm.groupby('seed').n_placebo.max().tolist()}")
print("\n== main methods (mean over seeds) ==")
print(allm[allm.method.isin(labels)].groupby("method")[cols].mean().reindex(labels).round(4).to_string())
print("\n== full grid: loop correction x outcome (mean over seeds) ==")
print(grid.round(4).to_string())
print("\n== mediation (gain, mean over seeds and chains) ==")
print(meds.round(4).to_string())

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL = {"Delta-spec only": "tab:gray", "SCM backdoor": "tab:orange", "SCM mediation": "tab:olive",
       "Potential outcome only": "tab:blue", "Potential outcome + Delta-spec": "crimson"}
short = {"Delta-spec only": "Δspec\nonly", "SCM backdoor": "SCM\nbackdoor",
         "SCM mediation": "SCM\nmediation", "Potential outcome only": "Potential\noutcome only",
         "Potential outcome + Delta-spec": "Potential outcome\n+ Δspec"}
fig, ax = plt.subplots(1, 4, figsize=(20, 4.8), gridspec_kw={"width_ratios": [1.6, 1, 1, 1]})

YLIM = (-2.5, 3.0)


def clip(v):
    v = np.asarray(v, float)
    return np.clip(v, YLIM[0] + 0.05, YLIM[1] - 0.05), (v < YLIM[0]) | (v > YLIM[1])


a = ax[0]
for i, lab in enumerate(labels):
    e = ests[ests.method == lab]
    p, off = clip(e[e.kind == "placebo"].relative_estimate)
    a.scatter(i + rng.uniform(-0.22, 0.22, len(p)), p, s=6, color="0.6", alpha=0.7,
              label="placebo chains" if i == 0 else None)
    if off.any():
        a.text(i, YLIM[1] - 0.25, f"{int(off.sum())} off scale", ha="center", fontsize=6, color="0.3")
    for c, mk in (("pc_OSK", "D"), ("pc_OSNL", "s")):
        q, qoff = clip(e[e.chain == c].relative_estimate)
        a.scatter([i + 0.3] * len(q), q, marker=mk, s=22, color="tab:green",
                  label=c.replace("pc_", "") if i == 0 else None)
    a.scatter([i] * len(seeds), [1.0] * len(seeds), marker="*", s=180, color=COL[lab], zorder=3,
              edgecolor="k", lw=0.5, label="OSKM" if i == 0 else None)
a.set_ylim(*YLIM)
a.axhline(0, color="0.4", lw=0.8)
a.axhline(1, color="0.4", lw=0.8, ls=":")
a.set_xticks(range(len(labels)), [short[x] for x in labels], fontsize=8)
a.set_ylabel("reported effect relative to OSKM's")
a.set_title("a  What each method reports\n(ideal: placebos at 0, OSKM at 1)")
a.legend(fontsize=7, loc="lower right")

b = ax[1]
mm = allm[allm.method.isin(labels)].groupby("method")
for lab in labels:
    g = mm.get_group(lab)
    b.errorbar(g.placebo_mean_ratio.mean(), g.z.mean(), xerr=g.placebo_mean_ratio.std(),
               yerr=g.z.std(), fmt="o", ms=9, color=COL[lab], capsize=3, label=short[lab].replace("\n", " "))
b.set_xlabel("placebo response / OSKM estimate\n(bias: share a random gene also gets)")
b.set_ylabel("separation z of OSKM from placebos")
b.set_title("b  Bias vs separation\n(best: upper left)")
b.axvline(0, color="0.6", lw=0.8)
b.legend(fontsize=7)

c = ax[2]
x = np.arange(len(labels))
top = mm.cell_top_frac.agg(["mean", "std"]).reindex(labels)
c.bar(x, top["mean"], yerr=top["std"], color=[COL[l] for l in labels], capsize=3)
c.set_xticks(x, [short[l] for l in labels], fontsize=7)
c.set_ylabel("share of cells where OSKM beats every placebo")
c.set_ylim(0, 1.05)
c.set_title("c  Per-cell detection")

dd = ax[3]
wdt = 0.38
for k, (cname, col) in enumerate((("pc_OSK_z", "tab:green"), ("pc_OSNL_z", "darkgreen"))):
    if cname not in allm:
        continue
    s = mm[cname].agg(["mean", "std"]).reindex(labels)
    dd.bar(x + (k - 0.5) * wdt, s["mean"], wdt, yerr=s["std"], color=col, capsize=2,
           label=cname.replace("pc_", "").replace("_z", ""))
dd.axhline(0, color="0.4", lw=0.8)
dd.axhline(1.96, color="0.4", lw=0.8, ls=":")
dd.set_xticks(x, [short[l] for l in labels], fontsize=7)
dd.set_ylabel("positive control z vs placebos")
dd.set_title("d  Known factor sets kept")
dd.legend(fontsize=7)
for axx in (c, dd):
    axx.tick_params(axis="x", labelsize=6.5)
fig.tight_layout()
fig.savefig(out / "methods.png", dpi=170)
fig.savefig(out / "methods.pdf")

# Supplementary: full grid, loop correction x outcome contrast
gm = allm[allm.method.isin([g[0] for g in GRID])].groupby(["loop", "outcome"])
panels = [("z", "separation z", "viridis"), ("cell_top_frac", "cells: OSKM beats every placebo", "viridis"),
          ("placebo_mean_ratio", "placebo response / OSKM estimate", "magma_r"),
          ("placebo_max_ratio", "largest placebo / OSKM estimate", "magma_r")]
loops = list(LOOPS)
fig2, ax2 = plt.subplots(1, len(panels), figsize=(18, 3.6))
for k, (col, title, cmap) in enumerate(panels):
    M = np.array([[gm.get_group((v, o))[col].mean() if (v, o) in gm.groups else np.nan
                   for o in ("raw", "dspec")] for v in loops])
    im = ax2[k].imshow(M, cmap=cmap, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax2[k].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="w" if im.norm(M[i, j]) < 0.5 else "k")
    ax2[k].set_xticks([0, 1], ["raw", "Δspec"])
    ax2[k].set_yticks(range(len(loops)), [LOOPS[v] for v in loops] if k == 0 else [""] * len(loops))
    ax2[k].set_title(title, fontsize=9)
fig2.suptitle("Loop correction (rows) x outcome contrast (columns), mean over seeds", fontsize=10)
fig2.tight_layout()
fig2.savefig(out / "methods_grid.png", dpi=170)
fig2.savefig(out / "methods_grid.pdf")
