#!/usr/bin/env python3
"""X7 causal screen (x7_causal.py): potential-outcome and structural-causal variants for separating the
OSKM-specific effect from the gene-independent (placebo) effect. Small-cell screen.

Perturbation chains (K -> M -> S -> O slots, the X5-R configuration):
  OSKM
  positive controls  pc_OSK (K,S,O), pc_OSNL (LIN28A,NANOG,SOX2,POU5F1; L and N take the K and M
                     slots), single factors pc_K, pc_M, pc_S, pc_O
  estimation placebos  est0..  (draw seed 1)  - only used to fit maps / generic mediator
  evaluation placebos  eval0.. (draw seed 0)  - negative controls, randomization inference
Each chain step i has a slot (0..3 = K, M, S, O); placebo genes are matched per slot.

Stage 1 (--mode stage1), decoder level. Per cell a fixed set of up to --genes-per-cell teacher
genes present in the start encoding; X = h_pert - h_ctrl (by token), Z = h_ctrl,
base = position in the perturbed encoding; the chain's own genes are excluded.
  V0      X
  V1      SCM backdoor adjustment for the gene's context: X - f(Z), f ridge on est placebos
  V1b     LEACE erasure of base rank fitted on est placebos
  V1+V1b  both
  V2      potential-outcome matched contrast: X - mean_est X for the same cell and gene
A decoder per (variant, chain): held-out-gene Spearman to the pluripotent teacher; the OSKM
decoder applied to every other chain. Saves OSKM decoders + maps to state.pt.

Stage 2 (--mode loop), feedback after every step with the OSKM decoders, per chain:
  ORE           Ordered rank-edit (no feedback)
  V0 V1 V1b V1+V1b   fixed-map variants
  med_generic   SCM mediator intervention: V0 decoder fed only the generic mediator m_g,
                m_g = mean over est placebos of the Delta-h obtained by swapping the chain's
                inserted genes for the placebo genes in the current encoding (same state,
                other gene identity)
  med_specific  V0 decoder fed X - m_g (the gene-specific mediator)
  V2            V2 decoder fed X - m_g
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

faulthandler.dump_traceback_later(int(os.environ.get("X7_TRACE_SEC", "1800")), repeat=True)

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["stage1", "loop"], required=True)
ap.add_argument("--code-root", type=Path, required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--max-ncells", type=int, default=50)
ap.add_argument("--genes-per-cell", type=int, default=256)
ap.add_argument("--n-est", type=int, default=10)
ap.add_argument("--n-eval", type=int, default=50)
ap.add_argument("--ridge", type=float, default=1e-2)
ap.add_argument("--chains", default="all", help="loop mode: comma list")
ap.add_argument("--variants", default="ORE,V0,V1,V1b,V1+V1b,med_generic,med_specific,V2")
ap.add_argument("--posctrl", default="pc_OSK,pc_OSNL,pc_K,pc_M,pc_S,pc_O")
ap.add_argument("--fit-max-rows", type=int, default=200000,
                help="stage1: rows from the est placebos used to fit the V1 / LEACE maps")
ap.add_argument("--tag", default=None)
args = ap.parse_args()

sys.path.insert(0, str(args.code_root / "core"))
from geneformer import in_silico_perturber as isp  # noqa: E402
from geneformer.auto_batch_size import coerce_batch_size  # noqa: E402
from geneformer.species_context import backend_from_config, default_isp_forward_batch_size  # noqa: E402
import run_ordered_rank_edit_isp as ore  # noqa: E402
import run_state_feedback_isp as rsf  # noqa: E402
from state_feedback import feedback as fb  # noqa: E402
from state_feedback import gene_states as gs  # noqa: E402
from state_feedback import random_chains  # noqa: E402
from state_feedback.decoder import (  # noqa: E402
    DeltaRankDecoder, TrainingSet, predict_delta_rank, train_delta_rank_decoder)
from state_feedback.metrics import spearman_values  # noqa: E402
from state_feedback.rerank import apply_priority_order, base_rank_norm  # noqa: E402
from state_feedback.teacher import observed_delta_rank, split_tokens  # noqa: E402

t0 = time.time()
torch.manual_seed(args.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out = args.out / f"seed{args.seed}"
out.mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load((args.code_root / "core/config/state_feedback_isp.yaml").read_text())
cfg["paths"]["dataset"], cfg["paths"]["geneformer_model"] = args.dataset, args.model
pert_cfg, sf_cfg = cfg["perturbation"], cfg["state_feedback"]
dec_cfg = dict(sf_cfg["decoder"])
state_key, start_state, goal_state = (pert_cfg["state_key"], pert_cfg["start_state"],
                                      pert_cfg["end_state"])

from datasets import load_from_disk  # noqa: E402

dataset = load_from_disk(args.dataset)
start_ds = ore._select_start_cells(dataset, state_key, start_state, args.max_ncells)
model = isp.load_model("CellClassifier", 2, args.model)
layer = isp.quant_layers(model) + int(cfg["isp"].get("emb_layer", 0))
mis = isp.get_model_input_size(model)
d = gs.hidden_size(model)
backend = backend_from_config(cfg)
with open(backend.token_dictionary, "rb") as fh:
    vocab = pickle.load(fh)
pad = vocab.get("<pad>")
fbs = ore.resolve_dual_forward_batch_size(
    coerce_batch_size("auto", default=default_isp_forward_batch_size(backend.max_input_size)),
    model, start_ds, pad, args.model)
B = max(1, int(fbs))
start_raw = gs.raw_input_ids(start_ds, 0, len(start_ds), mis)
print(f"mode {args.mode}; cells {len(start_ds)}; forward batch {fbs}", flush=True)


def steps_for(tbs):
    return [{"index": i + 1, "name": f"s{i + 1}", "type": "overexpress", "genes": [str(t[0])]}
            for i, t in enumerate(tbs)]


def kind_of(name):
    return "oskm" if name == "OSKM" else ("posctrl" if name.startswith("pc_") else
                                        ("est" if name.startswith("est") else "placebo"))


# =========================================================================================
if args.mode == "stage1":
    obs_cap = int(sf_cfg.get("observed_max_ncells") or 3000)
    obs_ds = ore._select_start_cells(dataset, state_key, goal_state, obs_cap)
    teacher_ctrl_ds = ore._select_start_cells(dataset, state_key, start_state, obs_cap)
    teacher = observed_delta_rank(teacher_ctrl_ds["input_ids"], obs_ds["input_ids"],
                                  min_detection_count=int(sf_cfg.get("min_detection_count", 5)))
    tok_names = {int(t): str(k) for k, t in vocab.items()}
    if getattr(backend, "gene_symbol_to_ensembl", None):
        with open(backend.gene_symbol_to_ensembl, "rb") as fh:
            e2s = {v: k for k, v in pickle.load(fh).items()}
        tok_names = {t: e2s.get(e, e) for t, e in tok_names.items()}

    def tok(g):
        return list(ore._resolve_gene_tokens(cfg, [g])[0])

    ORDER = ["KLF4", "MYC", "SOX2", "POU5F1"]
    conf_steps = [{"index": i + 1, "name": g, "type": "overexpress", "genes": [g]}
                  for i, g in enumerate(ORDER)]
    conf_tbs = [tok(g) for g in ORDER]
    perts = {"OSKM": conf_tbs}
    slots = {"OSKM": [0, 1, 2, 3]}
    POSCTRL = {"pc_OSK": (["KLF4", "SOX2", "POU5F1"], [0, 2, 3]),
               "pc_OSNL": (["LIN28A", "NANOG", "SOX2", "POU5F1"], [0, 1, 2, 3]),
               "pc_K": (["KLF4"], [0]), "pc_M": (["MYC"], [1]),
               "pc_S": (["SOX2"], [2]), "pc_O": (["POU5F1"], [3])}
    keep_pc = set(args.posctrl.split(",")) if args.posctrl else set()
    for name, (genes, sl) in POSCTRL.items():
        if name not in keep_pc:
            continue
        try:
            perts[name] = [tok(g) for g in genes]
            slots[name] = sl
        except Exception as exc:  # gene missing from the vocabulary
            print(f"skip {name}: {exc}", flush=True)
    population = sorted(int(t) for k, t in vocab.items() if not str(k).startswith("<"))
    det, rank = random_chains.start_profiles(start_raw)
    min_stratum = int(sf_cfg.get("stability", {}).get("min_stratum", 50))
    for rs, prefix, n in ((1, "est", args.n_est), (0, "eval", args.n_eval)):
        drawn, _ = random_chains.draw_random_chains(
            conf_steps, conf_tbs, population, det, rank, n_chains=max(20, n), seed=rs,
            min_stratum=min_stratum, prefix="r")
        for k in range(n):
            perts[f"{prefix}{k}"] = drawn[f"r{k}"][1]
            slots[f"{prefix}{k}"] = [0, 1, 2, 3]
    (out / "perturbations.json").write_text(json.dumps(
        {p: [tok_names.get(t[0], str(t[0])) for t in tbs] for p, tbs in perts.items()}, indent=1))
    EST = [p for p in perts if p.startswith("est")]

    rng = random.Random(args.seed)
    cell_idx, ctrl_pos, toks = [], [], []
    for i, ids in enumerate(start_raw):
        usable = [j for j, t in enumerate(ids) if int(t) in teacher]
        if len(usable) > args.genes_per_cell:
            usable = sorted(rng.sample(usable, args.genes_per_cell))
        for j in usable:
            cell_idx.append(i)
            ctrl_pos.append(j)
            toks.append(int(ids[j]))
    N = len(toks)
    target = torch.tensor([float(teacher[t]) for t in toks])
    train_tokens, val_tokens = split_tokens(
        teacher.keys(), val_fraction=float(dec_cfg.get("val_fraction", 0.2)),
        seed=int(dec_cfg.get("seed", 0)))
    is_train = torch.tensor([t in train_tokens for t in toks])
    is_val = torch.tensor([t in val_tokens for t in toks])
    rows_by_cell: dict[int, list[int]] = {}
    for r, c in enumerate(cell_idx):
        rows_by_cell.setdefault(c, []).append(r)

    def hidden(raw):
        return gs._hidden_states(model, raw, pad, layer).float()

    Z = torch.zeros((N, d))
    for lo in range(0, len(start_raw), B):
        H = hidden(start_raw[lo:lo + B])
        for c in range(lo, min(lo + B, len(start_raw))):
            rr = rows_by_cell.get(c, [])
            if rr:
                Z[rr] = H[c - lo, [ctrl_pos[r] for r in rr]].cpu()
        del H

    X, BASE, VALID = {}, {}, {}
    for p, tbs in perts.items():
        own = {int(t) for ts in tbs for t in ts}
        pds = rsf._apply_steps_at_once(start_ds, steps_for(tbs), tbs, nproc=1)
        praw = gs.raw_input_ids(pds, 0, len(pds), mis)
        x = torch.zeros((N, d))
        base = torch.zeros(N)
        valid = torch.zeros(N, dtype=torch.bool)
        for lo in range(0, len(praw), B):
            H = hidden(praw[lo:lo + B])
            for c in range(lo, min(lo + B, len(praw))):
                pos = {int(t): j for j, t in enumerate(praw[c])}
                br = base_rank_norm(len(praw[c]))
                for r in rows_by_cell.get(c, []):
                    j = pos.get(toks[r])
                    if j is None or toks[r] in own:
                        continue
                    x[r] = H[c - lo, j].cpu() - Z[r]
                    base[r] = br[j]
                    valid[r] = True
            del H
        X[p], BASE[p], VALID[p] = x.half(), base, valid
    print(f"features done ({len(perts)} chains, {N} samples) {time.time() - t0:.0f}s", flush=True)
    ore._empty_cuda_cache()

    def xf(p):
        return X[p].float()

    fit_gen = torch.Generator().manual_seed(args.seed)
    per_pert = max(1, args.fit_max_rows // len(EST))
    fit_rows = {}
    for p in EST:
        idx = VALID[p].nonzero().flatten()
        if len(idx) > per_pert:
            idx = idx[torch.randperm(len(idx), generator=fit_gen)[:per_pert]].sort().values
        fit_rows[p] = idx
    Xa = torch.cat([X[p][fit_rows[p]].float() for p in EST]).double()
    Za = torch.cat([Z[fit_rows[p]] for p in EST]).double()
    Ba = torch.cat([BASE[p][fit_rows[p]] for p in EST]).double()
    muX, muZ = Xa.mean(0), Za.mean(0)
    Zc = Za - muZ
    G = Zc.T @ Zc
    Bmat = torch.linalg.solve(G + args.ridge * torch.trace(G) / d * torch.eye(d, dtype=torch.float64),
                              Zc.T @ (Xa - muX))

    def fit_leace(xa, za):
        mu = xa.mean(0)
        xc = xa - mu
        cov = xc.T @ xc / xc.shape[0]
        ev, evec = torch.linalg.eigh(cov)
        ev = ev.clamp(min=ev.max() * 1e-8)
        W = evec @ torch.diag(ev.rsqrt()) @ evec.T
        Winv = evec @ torch.diag(ev.sqrt()) @ evec.T
        u = W @ (xc.T @ (za - za.mean()) / xc.shape[0])
        return mu, W @ (torch.outer(u, u) / (u @ u)) @ Winv

    maps = {"muX": muX, "muZ": muZ, "Bmat": Bmat}
    maps["leace_mu"], maps["leace_M"] = fit_leace(Xa, Ba)
    Xa1 = Xa - (muX + Zc @ Bmat)
    maps["leace1_mu"], maps["leace1_M"] = fit_leace(Xa1, Ba)
    del Xa, Za, Xa1, Zc

    est_valid = torch.stack([VALID[p] for p in EST]).float()
    n_est_valid = est_valid.sum(0)
    mean_est = sum(xf(p) * VALID[p].float().unsqueeze(-1) for p in EST) / n_est_valid.clamp(min=1).unsqueeze(-1)

    def T_v1(x, z):
        return (x.double() - (maps["muX"] + (z.double() - maps["muZ"]) @ maps["Bmat"])).float()

    def T_v1b(x):
        return (x.double() - (x.double() - maps["leace_mu"]) @ maps["leace_M"]).float()

    def T_v11b(x, z):
        y = T_v1(x, z).double()
        return (y - (y - maps["leace1_mu"]) @ maps["leace1_M"]).float()

    VARIANTS = {"V0": lambda p: xf(p), "V1": lambda p: T_v1(xf(p), Z),
                "V1b": lambda p: T_v1b(xf(p)), "V1+V1b": lambda p: T_v11b(xf(p), Z),
                "V2": lambda p: xf(p) - mean_est}
    grid = dec_cfg.get("max_shift_grid") or [0.1]

    def usable_mask(p, v):
        m = VALID[p].clone()
        return m & (n_est_valid > 0) if v == "V2" else m

    def train(F, m):
        tr, va = m & is_train, m & is_val
        ts = TrainingSet(delta_h=F[tr], base_rank=torch.zeros(int(tr.sum())), target=target[tr],
                         tokens=[toks[i] for i in tr.nonzero().flatten().tolist()])
        best = None
        for ms in grid:
            decd, _ = train_delta_rank_decoder(
                ts, d_model=d, max_shift=float(ms), epochs=int(dec_cfg.get("epochs", 20)),
                lr=float(dec_cfg.get("lr", 1e-3)), batch_size=int(dec_cfg.get("batch_size", 4096)),
                lam_huber=float(dec_cfg.get("lam_huber", 1.0)),
                lam_pair=float(dec_cfg.get("lam_pair", 0.5)),
                lam_smooth=float(dec_cfg.get("lam_smooth", 0.01)), seed=args.seed, device=dev)
            rho = spearman_values(predict_delta_rank(decd, F[va]).tolist(), target[va].tolist())
            if best is None or rho > best[0]:
                best = (rho, float(ms), decd)
        return best

    def rho_of(decd, F, m):
        va = m & is_val
        return spearman_values(predict_delta_rank(decd, F[va]).tolist(), target[va].tolist())

    rows, dec_state = [], {}
    others = [p for p in perts if p not in EST and p != "OSKM"]
    for v, fn in VARIANTS.items():
        rho, ms, decd = train(fn("OSKM"), usable_mask("OSKM", v))
        dec_state[v] = {"state_dict": {k: t.cpu() for k, t in decd.state_dict().items()},
                        "max_shift": ms}
        rows.append({"seed": args.seed, "variant": v, "chain": "OSKM", "kind": "oskm",
                     "decoder": "own", "rho_val": rho, "max_shift": ms})
        for p in others:
            F, m = fn(p), usable_mask(p, v)
            r_own, ms_own, _ = train(F, m)
            rows.append({"seed": args.seed, "variant": v, "chain": p, "kind": kind_of(p),
                         "decoder": "own", "rho_val": r_own, "max_shift": ms_own})
            rows.append({"seed": args.seed, "variant": v, "chain": p, "kind": kind_of(p),
                         "decoder": "oskm", "rho_val": rho_of(decd, F, m), "max_shift": ms})
        pd.DataFrame(rows).to_csv(out / "stage1_decoders.csv", index=False)
        print(f"stage1 {v}: OSKM rho {rho:.3f} ({time.time() - t0:.0f}s)", flush=True)

    def r2(x, xhat, m):
        x, xhat = x[m].double(), xhat[m].double()
        return float(1 - ((x - xhat) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())

    def ridge_rho(F):
        A = F[is_train].double()
        mu = A.mean(0)
        Gm = (A - mu).T @ (A - mu)
        y = target[is_train].double()
        w = torch.linalg.solve(Gm + args.ridge * torch.trace(Gm) / Gm.shape[0] * torch.eye(Gm.shape[0], dtype=torch.float64),
                               (A - mu).T @ (y - y.mean()))
        return spearman_values(((F[is_val].double() - mu) @ w).tolist(), target[is_val].tolist())

    fZ = (maps["muX"] + (Z.double() - maps["muZ"]) @ maps["Bmat"]).float()
    diag = []
    for p in others + ["OSKM"]:
        m = VALID[p] & (n_est_valid > 0)
        diag.append({"seed": args.seed, "chain": p, "kind": kind_of(p),
                     "r2_placebo_mean": r2(xf(p), mean_est, m), "r2_context": r2(xf(p), fZ, m),
                     "norm_ratio_contrast": float((xf(p)[m] - mean_est[m]).norm() / xf(p)[m].norm())})
    pd.DataFrame(diag).to_csv(out / "stage1_r2.csv", index=False)
    info = {"seed": args.seed, "n_samples": N, "n_cells": len(start_raw), "d": d,
            "teacher_from_context_Z_rho": ridge_rho(Z),
            "teacher_from_placebo_mean_dh_rho": ridge_rho(mean_est)}
    (out / "stage1_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info), flush=True)
    torch.save({"decoders": dec_state, "maps": {k: v.cpu() for k, v in maps.items()},
                "perts": perts, "slots": slots, "d": d}, out / "state.pt")

# =========================================================================================
else:
    state = torch.load(out / "state.pt", map_location="cpu", weights_only=False)
    perts, slots, maps = state["perts"], state["slots"], state["maps"]
    EST = [p for p in perts if p.startswith("est")]
    DEC = {}
    for v, s in state["decoders"].items():
        m = DeltaRankDecoder(d, max_shift=s["max_shift"]).to(dev)
        m.load_state_dict(s["state_dict"])
        m.eval()
        DEC[v] = m
    mapd = {k: v.to(dev) for k, v in maps.items()}
    centroid_ds = ore._centroid_dataset(dataset, state_key, [start_state, goal_state],
                                        args.max_ncells, 1)
    state_embs = isp.get_cell_state_avg_embs(
        model, centroid_ds, {"state_key": state_key, "start_state": start_state,
                             "goal_state": goal_state, "alt_states": []},
        layer, pad, fbs, ore._gpu_resident_map_workers(1))
    common = dict(layer_to_quant=layer, pad_token_id=pad, model_input_size=mis,
                  forward_batch_size=fbs, nproc=1)

    def T_fixed(v, x, z):
        if v == "V0":
            return x
        x = x.double()
        if v in ("V1", "V1+V1b"):
            x = x - (mapd["muX"] + (z.double() - mapd["muZ"]) @ mapd["Bmat"])
        if v == "V1b":
            x = x - (x - mapd["leace_mu"]) @ mapd["leace_M"]
        if v == "V1+V1b":
            x = x - (x - mapd["leace1_mu"]) @ mapd["leace1_M"]
        return x.float()

    def generic_mediator(pair, chain_toks, chain_slots):
        """m_g per row [L, d]: mean over est placebos of Delta-h after swapping the chain's
        inserted genes for the slot-matched placebo genes in the current encoding."""
        sums = [torch.zeros(len(ids), d, device=dev) for ids in pair.pert_ids]
        cnts = [torch.zeros(len(ids), device=dev) for ids in pair.pert_ids]
        ctrl_al = [gs.align_ctrl(ids, pair.ctrl_ids[r], pair.h_ctrl[r])
                   for r, ids in enumerate(pair.pert_ids)]
        for p in EST:
            ptoks = [int(perts[p][s][0]) for s in range(4)]
            swapped, excl = [], []
            for ids in pair.pert_ids:
                present = {int(t) for t in ids}
                sub = {c: ptoks[s] for c, s in zip(chain_toks, chain_slots) if c in present}
                repl = set(sub.values())
                new = [sub[int(t)] if int(t) in sub else int(t) for t in ids
                       if int(t) in sub or int(t) not in repl]
                swapped.append(new[:mis])
                excl.append(set(sub) | repl)
            H = gs._hidden_states(model, swapped, pad, layer).float()
            for r, ids in enumerate(pair.pert_ids):
                pos = {t: j for j, t in enumerate(swapped[r])}
                g = torch.tensor([pos.get(int(t), -1) if int(t) not in excl[r] else -1 for t in ids],
                                 device=dev)
                h_ctrl_al, valid = ctrl_al[r]
                ok = ((g >= 0) & valid.to(dev)).float().unsqueeze(-1)
                hp = H[r].index_select(0, g.clamp(min=0))
                sums[r] += (hp - h_ctrl_al.to(dev)) * ok
                cnts[r] += ok.squeeze(-1)
            del H
        return [s / c.clamp(min=1).unsqueeze(-1) for s, c in zip(sums, cnts)]

    def make_rerank(variant, chain):
        decd = DEC["V0" if variant.startswith("med_") else variant]
        need_mg = variant in ("med_generic", "med_specific", "V2")
        chain_toks = [int(t[0]) for t in perts[chain]]
        chain_slots = slots[chain]

        def fn(ctrl_ds, pert_ds):
            before, after = [], []
            for pair in gs.iter_gene_state_pairs(model, ctrl_ds, pert_ds, layer_to_quant=layer,
                                                 pad_token_id=pad, model_input_size=mis,
                                                 forward_batch_size=fbs):
                mg = generic_mediator(pair, chain_toks, chain_slots) if need_mg else None
                for row, ids in enumerate(pair.pert_ids):
                    h_pert, h_ctrl, valid = pair.aligned_states(row)
                    vm = valid.to(dev).float().unsqueeze(-1)
                    x = (h_pert.to(dev) - h_ctrl.to(dev)) * vm
                    if variant == "med_generic":
                        f = mg[row]
                    elif need_mg:
                        f = x - mg[row]
                    else:
                        f = T_fixed(variant, x, h_ctrl.to(dev))
                    base = torch.tensor(base_rank_norm(len(ids)), dtype=torch.float32, device=dev)
                    with torch.no_grad():
                        pr = decd(f * vm, base).cpu().tolist()
                    before.append(list(ids))
                    after.append(apply_priority_order(ids, pr, hysteresis=0.0))
            return after, fb.rerank_diagnostics(before, after)
        return fn

    def final_cells(dirp: Path) -> pd.Series:
        n = len(perts[dirp.name]) if dirp.name in perts else 4
        st = sorted(q for q in dirp.iterdir() if q.is_dir() and q.name.startswith(f"step{n:02d}"))
        fed = [q for q in st if q.name.endswith("_feedback")]
        return pd.read_csv((fed or st)[0] / "per_cell_shifts.csv").set_index("cell_index")["Shift_to_goal_end"]

    chains = [c for c in perts if c not in EST] if args.chains == "all" else args.chains.split(",")
    variants = args.variants.split(",")
    tag = args.tag or f"{chains[0]}-{chains[-1]}"
    rows = []
    done_csv = out / f"stage2_per_cell_{tag}.csv"
    if done_csv.exists():
        prev = pd.read_csv(done_csv)
        complete = {c for c, g in prev.groupby("chain") if set(variants) <= set(g.variant)}
        rows = prev[prev.chain.isin(complete)].to_dict("records")
        print(f"resume: {sorted(complete)} already done", flush=True)
        chains = [c for c in chains if c not in complete]
    for c in chains:
        ore_dir = out / "loop" / "ORE" / c
        rsf.run_condition(f"ORE_{c}", model, start_ds, steps_for(perts[c]), perts[c], goal_state,
                          state_embs, ore_dir, rerank_fn=None, **common)
        s_ore = final_cells(ore_dir)
        for v in variants:
            if v == "ORE":
                s = s_ore
            else:
                dirp = out / "loop" / v / c
                t1 = time.time()
                rsf.run_condition(f"{v}_{c}", model, start_ds, steps_for(perts[c]), perts[c],
                                  goal_state, state_embs, dirp, rerank_fn=make_rerank(v, c),
                                  pin_overexpressed=True, ctrl_reference="start", **common)
                s = final_cells(dirp)
                print(f"loop {v} / {c}: median {s.median():.4f} gain "
                      f"{(s - s_ore.loc[s.index]).mean():.4f} ({time.time() - t1:.0f}s)", flush=True)
            rows += [{"seed": args.seed, "variant": v, "chain": c, "kind": kind_of(c),
                      "cell_index": int(i), "endpoint": float(s.loc[i]),
                      "ore": float(s_ore.loc[i]), "gain": float(s.loc[i] - s_ore.loc[i])}
                     for i in s.index]
        pd.DataFrame(rows).to_csv(out / f"stage2_per_cell_{tag}.csv", index=False)

print(f"done in {time.time() - t0:.0f}s", flush=True)
