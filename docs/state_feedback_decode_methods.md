# State-feedback ISP v10.1 — permutation decoder 設計

## 問題の正確な定義

State-feedback ISP は「hidden state から発現量を復元する」問題**ではない**。

Geneformer が受け取る **相対的な遺伝子順位を、摂動後の状態から一貫して生成する問題** — すなわち **permutation decoder** として定義する。

```text
Xt   = (g_{t,1}, ..., g_{t,n})     現在の gene-token 順位列
Ht   = F_θ(Xt; OE_t)              摂動後 hidden states
Ht⁰  = F_θ(Xt)                    非摂動 hidden states (同一入力)
st   = D_ϕ(Ht, Ht⁰, Xt)          per-gene scalar score
Xt+1 = Sort↓(Xt, st)              reranked token list (同一遺伝子集合、順序のみ置換)
```

**Key constraint**: 遺伝子集合は固定。Token の挿入/削除を許すと generative transcriptome completion になり、別の問題。State-feedback ISP v10.1 は **fixed detected-gene set 上の conditional reranking**。


## FT-mandatory premise

ISP Platform では Geneformer を cell-state classifier として fine-tune しなければ ISP 結果は信頼できない。Pre-trained model の出力はほぼ無意味。この前提は decoder 選択に決定的な影響を持つ:

- FT 後の hidden states は **classification objective に最適化された表現空間** にある
- Pre-trained MLM head は FT 中に更新されないため、FT'd representations とは **domain mismatch** を起こす
- ISP signal = FT'd 空間における `h_pert - h_ctrl` であり、decoder はこの空間に対して整合していなければならない


## 手法の役割分担

| 手法 | 役割 | FT-aware rationale |
|:---|:---|:---|
| **Residual Δrank decoder** | **主法** | FT'd 表現空間上で専用に学習。delta-h を直接入力とし、表現空間と decoder が整合する唯一の手法 |
| ΔMLM self-logit + inertia | Parameter-free baseline | Pre-trained MLM head は FT 後 degraded だが、pert - ctrl 差分でバイアスがキャンセル。学習不要の comparator として有用 |
| MLM absolute self-logit | **Not recommended** | Pre-trained head × FT'd representations = domain mismatch。FT 必須の ISP Platform では本質的に壊れている |
| Embedding norm | Null baseline / sanity check | 学習構造なし。norm ≠ rank |
| MLP Δrank decoder | 上限比較 | Linear が飽和した場合のみ |
| Cosine-to-input-emb | 探索的 | 入出力 embedding 空間の整合性未保証; FT でさらに乖離 |
| Attention centrality | **主張から除外** | Attention ≠ importance (Jain & Wallace 2019) |

**なぜ absolute MLM は壊れるのに ΔMLM は使えるか:** 差分 (pert - ctrl) では同一の degraded head を通すため系統的バイアスがキャンセルされる。absolute MLM は壊れた head の出力をそのまま読むため、FT 後は意味をなさない。ただし ΔMLM も「token plausibility change」と「expression-rank change」が等価でない点は変わらないため、primary decoder にはしない。


## Residual Δrank decoder — 最小実装

```python
class DeltaRankDecoder(nn.Module):
    def __init__(self, d_model: int, max_shift: float):
        super().__init__()
        self.proj = nn.Linear(d_model + 1, 1)
        self.max_shift = max_shift

    def forward(self, h_pert, h_ctrl, base_rank_norm):
        delta_h = h_pert - h_ctrl          # 摂動効果のみ
        x = torch.cat([delta_h, base_rank_norm.unsqueeze(-1)], dim=-1)
        delta_rank = self.max_shift * torch.tanh(self.proj(x).squeeze(-1))
        return base_rank_norm + delta_rank  # argsort → new input_ids
```

**入力設計の意図**: `delta_h`（`h_ctrl` ではなく差分）を使うことで、decoder が base cell identity ではなく **perturbation effect** を読むよう制約。`h_ctrl` の追加は後続 ablation。

**`tanh` bound の意図**: 1 step で bottom → top のような病的ランクジャンプを抑止。`max_shift` は {0.05n, 0.10n, 0.20n} で grid search。


## Oracle-first 検証（decoder 学習前に実施必須）

```text
Oracle: 観測された摂動後ランクリストを next-step input_ids として使用
```

| 結果 | 解釈 |
|---|---|
| Oracle で 2nd-step ISP 改善なし | **State-feedback 自体に価値なし** → decoder 改善は無意味 |
| Oracle で改善あり & trained decoder が Oracle に接近 | 研究仮説が支持される |


## 実装ガードレール（v1.0.1 必須）

1. **Fixed gene set** — trajectory 内で token の挿入/削除なし
2. **Bounded displacement** — tanh / clamp; max_shift を grid search
3. **Null-drift check** — null perturbation → Δr ≈ 0 を検証、drift 量を報告
4. **Swap hysteresis** — |ŝ_i − ŝ_j| < ε の隣接ランクは元の順序を保持
5. **Cycle detection** — X_t ↔ X_{t+1} の 2-cycle を検出、halt
6. **Convergence stopping** — Spearman(r_t, r_{t+1}) > 0.995 × 2 step、または top-1000 Jaccard > 0.99、または hard cap T ≤ 5
7. **Cell-wise normalization** — ランクは cell-specific 相対量; 異なる sequence length の細胞間でスコアを直接比較しない


## 実装戦略（gated）

**推奨:** Phase 0 は hard gate。Phase 1–2 はまとめて実装してよい。Phase 3 まで一気に実装しない。

**Hard gate は 2 箇所:** Phase 0 完了後 と Phase 2 完了後。

| Phase | やること | 止まるか？ | 理由 |
|:---:|:---|:---:|:---|
| **0** | Oracle feedback 実験 | **ここで必ず止まって結果を見る** | Oracle で改善なし → 以降すべて無意味。Go/no-go ゲート |
| **1** | Norm / ΔMLM baselines | → 2 と一緒に実装して OK | Parameter-free で実装コスト小。2 との比較用 |
| **2** | Linear Δrank decoder | **ここで止まって結果を見る** | 主法。1 との差・Oracle との gap を確認。生物学的検証はここで |
| **3** | MLP Δrank / multi-step | 2 の結果次第 | Linear が oracle gap の 80% 以上を達成していれば不要 |

### なぜこの順か

1. **Phase 0 をスキップして Phase 2 を先に作ると無駄になる。** State-feedback に価値が無い場合、decoder 実装はすべて無意味。Oracle は既存 ISP に observed rank を食わせるだけなので、decoder より先にできる。
2. **Phase 1 と 2 を分けて「1 の結果を見てから 2」にする必要はない。** 1 は parameter-free で安い。2 の比較対象として必要なだけなので、同時実装・同時評価が効率的。
3. **Phase 2 の結果で立ち止まる。** 見るべきは:
   - Linear Δrank vs ΔMLM vs Oracle の gap
   - Predicted Δrank vs observed Δrank の Spearman ρ
   - Top perturbed genes の precision@K
   - 既知 pathway / enrichment との一致
   - Null perturbation での drift
4. **Phase 3 は Phase 2 が頭打ちの場合のみ。** Linear で oracle の ~80% 以上なら MLP 不要。Multi-step stability は Linear が動いてから。

### Phase 0 data pin (this host)

- Analysis: `~/20260624Geneformer-Platform` · **BBRCv3** (or later)
- Workspace copy: `data/bbrcv3/tokenize_human_native`, `data/bbrcv3/v2_ft_human_native` (`/data/` is gitignored)
- Runner: `python3 core/run_state_feedback_oracle.py --config core/config/state_feedback_oracle.yaml`
- Oracle surrogate on OSKM endpoints: start = somatic token ranks; observed post = pluripotent token ranks projected onto each cell’s **fixed gene set** (permutation only)

### Phase 0 status

- Helpers + tests: `core/state_feedback/oracle_rerank.py`, `tests/test_state_feedback_oracle.py` (5/5 OK)
- Smoke (n=50, K-M-S-O, 2026-09-26): ordered_rank_edit final median **0.0125** vs oracle_mid **0.640** vs oracle_endpoint **0.601** → provisional GO
- **Next:** confirm gate at `isp.max_ncells: 300` (config default) before Phase 1–2; do not start decoder work until that CSV is reviewed


### Phase 1–2 status (2026-09-26)

Implemented, generic (no OSKM hardcoding anywhere in `core/state_feedback/`):

| File | Role |
| --- | --- |
| `core/state_feedback/gene_states.py` | streaming gene-level hidden states; token-identity alignment; MLM self-logits |
| `core/state_feedback/rerank.py` | priority convention, bounded shift, swap hysteresis, `norm` / `delta_mlm` scorers |
| `core/state_feedback/teacher.py` | observed Δrank from pseudobulk positions; gene-level train/val split |
| `core/state_feedback/decoder.py` | `DeltaRankDecoder`, training loop, `max_shift` grid search, null drift |
| `core/state_feedback/feedback.py` | rerank strategies + training-sample collection |
| `core/run_state_feedback_isp.py` | runner comparing all conditions in one run |
| `core/config/state_feedback_isp.yaml` | config (OSKM values are a worked example only) |
| `tests/test_state_feedback_phase12.py` | 34 tests (39 with Phase 0) — all pass in Docker |

**How to run** (host `spark-943a`, Docker):

```bash
ISP_PLATFORM_IMAGE=isp-platform:latest DOCKER_UID=$(id -u) DOCKER_GID=$(id -g) \
  docker compose run --rm --no-deps sequential_isp \
  python3 core/run_state_feedback_isp.py --config core/config/state_feedback_isp.yaml --max-ncells 50
# tests
ISP_PLATFORM_IMAGE=isp-platform:latest DOCKER_UID=$(id -u) DOCKER_GID=$(id -g) \
  docker compose run --rm --no-deps sequential_isp \
  python3 -m unittest tests.test_state_feedback_phase12 tests.test_state_feedback_oracle
```

`--conditions` selects a subset (conditions are mutually independent, so they can be
split across jobs). Wall time at n=50 was ~26 min end to end; ~4 min per condition,
dominated by GPU forward passes and therefore roughly linear in cell count.

**Genericity.** The perturbation is whatever `sequential.steps` lists (any genes,
`overexpress` or `knockdown`, any number of steps); states are whatever
`perturbation.state_key / start_state / end_state` name. Switching to another
fine-tuned model and another cell-state pair requires only config edits.

Requirements this imposes on a user dataset (both are already implied by ISP itself,
since the goal-state centroid needs observed goal cells):

1. start and goal cells in the **same** tokenized dataset under the same `state_key`
2. enough goal cells for a stable pseudobulk teacher (n=50 run: 3000 observed cells →
   12,254 usable teacher tokens at `min_detection_count: 5`)

`state_feedback.observed_state` exists so the teacher can be an observed *intermediate*
state rather than the goal; it falls back to `end_state`.

#### Smoke results, n=50, OSKM example (`state_feedback_isp_072807`)

| Condition | mean abs displacement | shift after feedback | final median | gap closed |
| --- | --- | --- | --- | --- |
| `ordered_rank_edit` | 0 | — | 0.0125 | baseline |
| `linear_deltarank` | 29.7 | 0.1276 | 0.1408 | 0.205 |
| `delta_mlm` | 216.4 | 0.1348 | 0.1492 | 0.218 |
| `norm` (null baseline) | 226.2 | 0.1279 | 0.1433 | 0.208 |
| `oracle` (ceiling) | 639.0 | 0.6012 | 0.6396 | 1.0 |
| `null_feedback` (control) | 0.0 | — | 0.0000 | — |

Decoder, held-out **genes** (12,254 teacher tokens → 9,808 train / 2,446 val):

| `max_shift` | Spearman train | Spearman val | null drift (rank units) |
| --- | --- | --- | --- |
| 0.05 | 0.327 | 0.316 | 3.1 |
| 0.10 | 0.362 | 0.359 | 5.7 |
| 0.20 (selected) | 0.398 | 0.400 | 8.4 |

#### Phase 2 gate: **INCONCLUSIVE — do not proceed to Phase 3**

Guardrails passed. `null_feedback` moved zero genes (displacement 0.0, Spearman 1.0,
top-100 Jaccard 1.0, shift 0.0000): with no perturbation evidence the decoder is exactly
the identity permutation. The analytic null drift is non-zero (0.004 normalized) but is a
monotone function of `base_rank_norm` and therefore order-preserving.

The gate metric cannot discriminate. `norm` — deliberately theory-free, included to lose —
closes 20.8% of the oracle gap against the decoder's 20.5%. Three observations say this is
a property of the metric, not of the decoder:

1. all three feedback conditions jump to ~0.13 at the feedback step and then plateau
2. displacement differs 7-fold (29.7 vs 216 vs 226) while shift barely moves (0.141 / 0.149 / 0.143)
3. `null_feedback` gives exactly 0.0, so zero displacement does give zero shift

The cell-mean cosine shift therefore responds to *whether* the encoding was disturbed far
more than to *which direction* it was permuted — in a two-class space "away from somatic"
reads as "toward pluripotent". `oracle` escapes the plateau because pasting observed
pluripotent ranks makes the encoding a pluripotent cell, which is close to tautological.

Two signals do favour the decoder, neither decisive: Spearman 0.400 against observed Δrank
on held-out **genes** (train 0.398 — no overfitting, as expected for 769 parameters), and
reaching the same plateau with 29.7 displacement where `norm` needs 226 (7.6× more
efficient per unit of displacement).

**Missing measurement that would settle it.** `norm` and `delta_mlm` have no Spearman
against the same teacher on the same held-out gene split. Without it, "does the learned
decoder beat the theory-free baseline" is unanswered. This is direction-sensitive and
independent of the degenerate shift metric.

Phase 3 is **not** the right response: the eval is confounded, not failed. Enlarging the
model cannot fix a metric that ignores permutation direction.


## 必須比較条件

| Condition | Description |
| --- | --- |
| Ordered rank-edit (current) | Paper baseline; token-edit chain, no embedding feedback |
| No-feedback | 1st step rank list frozen for all subsequent steps |
| Norm reranking | Null baseline |
| ΔMLM + inertia | Parameter-free differential baseline |
| Linear residual Δrank | Primary method |
| MLP residual Δrank | Upper-bound comparator (if linear saturates) |
| **Oracle reranking** | Observed perturbation rank → performance ceiling |


## Multi-step stability metrics (T ≥ 2 で報告)

- Per-step rank correlation: Spearman(X_t, X_{t+1})
- Top-K overlap (Jaccard) at K = 100, 500, 1000
- Rank displacement distribution per step
- Null-perturbation cumulative drift
- 2-cycle / oscillation frequency
- Multi-seed reproducibility
- Convergence step count distribution


## Claims boundary for Methods

**Safe to claim:**
> State-feedback ISP uses a learned residual rank decoder that maps perturbation-induced changes in contextual gene representations to bounded gene-specific rank displacements. The decoder preserves the detected gene set and updates only the within-cell ordering. Parameters were trained and evaluated against observed perturbation-associated rank shifts on held-out datasets.

**Do not claim:**
- "Hidden-state norm represents gene expression"
- "MLM logits recover expression values"
- "Attention weights quantify gene importance"
- "Decoded list is a simulated RNA-seq measurement"
- "Iterated output is a causal cellular trajectory" (requires wet-lab time-series validation)


## Training data & loss

**Teacher signal:** observed rank displacement from paired perturbation data.

```text
Δr_i^obs = r_i^{pert,obs} − r_i^{ctrl,obs}
```

Use **pseudobulk / metacell ranks** (not single-cell raw ranks) to suppress dropout noise.

Data sources (priority order):
1. CRISPRi / CRISPRa Perturb-seq
2. Gene OE Perturb-seq
3. Drug treatment pre/post scRNA-seq
4. Isogenic OE / KD experiments
5. Time-series reprogramming / differentiation datasets

**Loss (initial):**

```text
L = λ₁ · Huber(Δr_pred, Δr_obs)
  + λ₂ · L_pair           (pairwise / ListMLE ordering loss)
  + λ₃ · L_identity        (ctrl→ctrl: Δr ≈ 0)
  + λ₄ · L_smooth          (regularize extreme Δr)
```


## scPRINT-2 reference (design parallel, not drop-in)

Found under `~/20260323scPRINT-2/` (also `~/20260531scPRINT2andPanOrganseq/scPRINT-2-private/`):

- Spec: `docs/ISP_v2_spec.md`
- `predict_expression` → `pred_start.h5ad` / `pred_pert.h5ad` / `delta_ranked_genes.csv`
- Enrichment: `services/run_isp_enrichment.py` (gseapy prerank)

scPRINT predicts expression via ZINB mean (`scprint_mu`), then ranks deltas. This is analogous to MLP Δrank + expression decoder — natural for scPRINT's architecture but inapplicable to Geneformer (no expression decoder). The residual Δrank approach above achieves a comparable pipeline shape without requiring an expression decoder.
