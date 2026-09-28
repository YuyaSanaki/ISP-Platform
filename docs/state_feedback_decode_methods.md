# State-feedback ISP v10.1 — permutation decoder 設計

User-facing guide (how it works, conditions, Web UI settings): [ordered_rank_edit_and_state_feedback_isp.md](ordered_rank_edit_and_state_feedback_isp.md) (v1.0.1). This page is the design and validation record.

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
| Embedding norm | **Unsupervised scalar baseline** | 学習構造なし。norm ≠ rank。詳細は下記専用セクション |
| MLP Δrank decoder | 上限比較 | Linear が飽和した場合のみ |
| Cosine-to-input-emb | 探索的 | 入出力 embedding 空間の整合性未保証; FT でさらに乖離 |
| Attention centrality | **主張から除外** | Attention ≠ importance (Jain & Wallace 2019) |

**なぜ absolute MLM は使わず ΔMLM を comparator にするか:** absolute MLM は壊れた head の出力をそのまま読むため FT 後は意味をなさない。差分 (pert − ctrl) は同一 head を通すので **共有された calibration offset を緩和することが期待される**。ただしこれは仮説であって保証ではない — FT による hidden-state geometry の変化は perturbation 依存であり、同じ frozen head を通しても非線形性や calibration failure が差分で消えない場合がある。加えて ΔMLM は「token plausibility change」であって「expression-rank change」ではない。したがって ΔMLM は良い parameter-free comparator だが、**domain mismatch が数学的に相殺されると主張してはならない**。

Methods 向け推奨表現:

> Differential self-logits may attenuate shared calibration offsets relative to absolute self-logits, but they remain a token-plausibility proxy and require empirical validation against observed rank displacement.

実測 (n=50, held-out genes): ΔMLM の観測 Δrank に対する Spearman は **−0.016**（pooled）で、chance と区別できない。この dataset では「緩和が起きて rank 情報が残る」という期待は**支持されなかった**。


## Embedding-norm baseline — 何を使い、何のために置くか

実装: `core/state_feedback/feedback.py::rerank_norm` + `core/state_feedback/rerank.py::norm_priority`

### 何を使っているか

同一細胞の2つの encoding から取った**遺伝子ごとの hidden state ベクトル**のみ。`h_ctrl` は非摂動 encoding、`h_pert` は摂動後 encoding における当該 gene-token の hidden state。両 encoding は位置を共有しないので **token 同一性**で対応付ける（`gene_states.align_ctrl`）。スコアは **L2 ノルムの差** という単一スカラー。

```python
delta_norm = ‖h_pert[gene]‖₂ − ‖h_ctrl[gene]‖₂     # feedback.py
shift      = −alpha · zscore_within_cell(delta_norm)
priority   = base_rank_norm + clamp(shift, ±max_shift)   # rerank.py
```

`priority` は昇順＝左端＝高ランクの規約なので、**ノルムが増えた遺伝子が左（高発現ランク側）へ動く**。変位は `±max_shift` でクランプされ、`base_rank_norm`（元順位）が慣性項として残る。

### 何のために置くか

**parameter-free unsupervised reranking baseline。** 論拠:

1. Geneformer には expression decoder が無いので、hidden state から「発現がどれだけ変わったか」を読む試みはすべて**推測**である。
2. その中で最も安易かつ single-cell 基盤モデル文脈で頻繁に提案されるのが「embedding が大きい＝重要／高発現」という読み方である。
3. これには **expression decoder としての理論的根拠が無い**。transformer の hidden-state ノルムは token 頻度・layer norm スケール・位置文脈・attention sink 的挙動を反映しており、発現順位を表すように作られていない。
4. ゆえに「学習を伴わないスカラー heuristic で並べ替える」条件として置く。**これに明確に勝てない限り、decoder が実在の発現順位シグナルを読めている証拠にならない。**

**なぜランダム置換ではなくノルムか。** ランダム置換は rank 情報を一切持たないため「勝つのが簡単すぎる」（`random` / `identity` として別途 null に置く）。`norm` は decoder と機械部分（遺伝子ごとの hidden state、同じ bounded shift、同じ swap hysteresis、同じ base-rank 慣性）を共有し、**異なるのは読むスカラーだけ**。変数を1つに絞った締まった比較になる。

### 呼称の注意（Methods 向け）

内部文書では「陰性対照」「負けるために置いた」で構わないが、**論文では強すぎる**。hidden-state norm は expression decoder ではない一方、position embedding・residual stream・layer normalization・context-dependent activation を介して rank と相関しうる。強い null と断定して実測で強く出た場合に矛盾する。

推奨表現:

> We used the within-cell change in hidden-state L2 norm as a parameter-free unsupervised reranking baseline. This baseline shares the same token alignment, bounded-displacement rule, rank inertia, and hysteresis mechanism as the learned decoder, but replaces the learned mapping with a scalar representation heuristic. It was not interpreted as a measure of gene expression.

この表現なら (a) norm を発現量と同一視せず、(b) 同一 reranking mechanics を共有した公正な比較であることを示し、(c) norm が強く出た場合も矛盾せず、(d) 後から neutral に再定義できる。

### 結果の解釈

| パターン | 意味 |
|---|---|
| decoder ≫ norm（direction fidelity で） | 学習した decode が実在の順位情報を運んでいる。**Phase 2 direction 軸 合格** |
| decoder ≈ norm（endpoint shift のみで） | 指標が両者を区別できていない可能性が第一候補。direction fidelity で再判定する |
| norm ≈ decoder（direction fidelity でも） | `norm` は null ではない。v10.1 の失敗ではなく、hidden-state norm に position / rank / context の実在シグナルが漏れていることを意味する。その場合 “negative control” の呼称をやめ **nonparametric representation baseline** と呼び替える |

**実測 (n=50, held-out genes, 観測 Δrank に対する pooled Spearman):**

| Method | pooled ρ | cell-wise median ρ | gene-aggregated ρ |
|---|---|---|---|
| `linear_deltarank` | **+0.419** | **+0.443** | **+0.613** |
| `norm` | +0.028 | +0.025 | +0.149 |
| `delta_mlm` | −0.016 | −0.002 | −0.038 |
| `random` | −0.002 | −0.004 | −0.020 |
| `identity` | 未定義（定数予測なので分散ゼロ） | — | — |

この dataset では `norm` は**実質的に無情報**（pooled ρ = 0.028）であり、endpoint shift で decoder と同着だったのは**指標側の縮退が原因だったことが確定した**。したがってここでは呼称を neutral にしたまま、「norm は rank 情報を持たなかった」と実測として報告できる。ただしこれは本 dataset での観察であり、他 dataset で norm が強く出る可能性は残る。

### 採点時の注意（clamp を掛けてはならない）

direction fidelity の採点は **bounded displacement を適用する前のスコア**で行う。clamp は配線上のガードレールであって採点関数ではなく、z-score baseline に対して **rank を保存しない**。`alpha = 1`, `max_shift = 0.1` では単位正規 z の約 92% が境界外に出るため、baseline が実質2値の符号予測に潰れて decoder に対して不当に不利になる。decoder 側の `tanh` は pre-activation に対して単調なので `max_shift` を外しても順位は不変で、比較は対称に保たれる（`evaluate.predicted_delta_rank(bound=False)`）。


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

**`tanh` bound の意図**: 1 step で bottom → top のような病的ランクジャンプを抑止。`max_shift` は grid search（実装では val Spearman が 0.20 で伸び続けたため {0.05, 0.1, 0.2, 0.3, 0.5} に拡張）。

**実装との対応** (`core/state_feedback/decoder.py`): 上記 `forward(h_pert, h_ctrl, base_rank_norm)` は `from_states()` として保持し、中核は `delta_h` を直接受ける `delta_rank()` / `forward(delta_h, base_rank_norm)`。同じ `delta_h` を学習と推論で再利用するため。`proj` は **zero-init** なので未学習 decoder は厳密に恒等置換になる。

**pairwise loss の適用先に注意.** 順序損失は **変位 (Δrank) に対して**掛ける。絶対 priority に対して掛けると `base_rank_norm` を tanh で増幅するだけで最小化でき（= 構成上すでに正しい順序に変位予算を全部使う）、Δh シグナルが消える。合成線形教師での Spearman が 0.999 → 0.215 に落ちることで確認済み。


## Oracle-first 検証（decoder 学習前に実施必須）

```text
Oracle: 観測された摂動後ランクリストを next-step input_ids として使用
```

| 結果 | 解釈 |
|---|---|
| Oracle で 2nd-step ISP 改善なし | **State-feedback 自体に価値なし** → decoder 改善は無意味 |
| Oracle で改善あり & trained decoder が Oracle に接近 | 研究仮説が支持される |

### Oracle は 2 種類に分ける

**Oracle-A: Endpoint rank oracle**（実装済み = 現行 `oracle` 条件）

- 入力: 観測された goal-state（例: pluripotent）の rank
- 用途: 理論上限、endpoint classifier が rank reranking に応答するかの確認
- **注意: goal-state 情報を次入力に直接入れているため endpoint leakage を含む。** 貼った時点で encoding 自体が goal 細胞に近づくので改善はほぼ同義反復。**decoder の性能目標として使ってはならない。**

**Oracle-B: Matched perturbation oracle**（未実装 — データ待ち）

- 入力: **同一 perturbation の実測 post-perturbation rank**（例: OSKM 導入後の適切な時点の観測 rank）
- 用途: 「実際に起きた 1st perturbation response を次 ISP に戻す」こと自体に価値があるかの判定
- 長所: decoder が目指す Δr_obs と**同じ target** を使うので、leakage なく fidelity 目標になる
- 時間分解版:
  ```text
  X_somatic → X_OSKM,t1^obs → ISP_2
  X_somatic → X_OSKM,t2^obs → ISP_3
  ```

**現 dataset では Oracle-B は構成できない。** `GSE147564_ft6k_qc_human_native_0` は `time ∈ {D0, P20P}` × `genotype ∈ {somatic, pluripotent}` の 2 群のみで、`time` / `genotype` / `disease` / `sample_id` は互いに完全に共線、`replicate` は `Rep1` 単一。OSKM 誘導後の**中間時点が存在しない**ため、matched post-perturbation rank が取れない。Oracle-B には reprogramming time-course dataset（day 3 / 7 / 14 等）の追加が必要。


## 実装ガードレール（v1.0.1 必須）

| # | ガードレール | 内容 | 実装状況 |
|:---:|:---|:---|:---|
| 1 | **Fixed gene set** | trajectory 内で token の挿入/削除なし | 済 — `rerank.apply_priority_order` は置換のみ（`test_gene_set_is_preserved`） |
| 2 | **Bounded displacement** | tanh / clamp; max_shift を grid search | 済 — `decoder.delta_rank` の `max_shift·tanh`、baseline は `rerank.bounded_priority` |
| 3 | **Null-drift check** | null perturbation → Δr ≈ 0 を検証、drift 量を報告 | 済 — 解析値 `decoder.null_drift` + 実走条件 `null_feedback`。実測は**変位 0.0 / Spearman 1.0 / shift 0.0000**（解析 drift 0.004 は `base_rank_norm` の単調関数なので順序不変） |
| 4 | **Swap hysteresis** | \|ŝ_i − ŝ_j\| < ε の隣接ランクは元の順序を保持 | 済 — priority を ε グリッドに量子化してから**安定ソート**（近接差を厳密な同値に変える）。既定 `hysteresis: 0.0`（無効） |
| 5 | **Cycle detection** | X_t ↔ X_{t+1} の 2-cycle を検出、halt | 済 — `multistep.FeedbackGuard`。細胞ごとに、提案された post_t が post_{t−2} と一致し post_{t−1} と異なれば（`metrics.is_two_cycle`）提案を棄却して pre_t を保持し、その細胞を halt |
| 6 | **Convergence stopping** | Spearman(r_t, r_{t+1}) > 0.995 × 2 step、または top-1000 Jaccard > 0.99、または hard cap T ≤ 5 | 済 — `multistep.FeedbackGuard`。feedback 1 回の変化 Spearman(pre_t, post_t) > 0.995、または top-K Jaccard > 0.99 が `converge_patience`（既定 2）回連続した細胞を停止。hard cap は chain あたり `max_feedback_events`（既定 5） |
| 7 | **Cell-wise normalization** | ランクは cell-specific 相対量; 異なる sequence length の細胞間でスコアを直接比較しない | 済 — `base_rank_norm = i/(n−1)` で細胞内正規化、z-score も細胞内。教師 Δrank も各 state の平均 encoding 長で正規化 |

5 と 6 は `feedback_every_step: true`（`feedback_after_step` 以降、最終 step を除く各 step の後に feedback）のときだけ働く。既定の feedback 1 回では発火しない。設定は `state_feedback.multi_step` に置く。

- **比較するのは同じ feedback の前後**。連続する post encoding どうしは、間に摂動 step が入って動くので直接比べない。
- **Jaccard 基準は細胞の遺伝子数が K を超えるときだけ使う**。gene set が固定なので、遺伝子数が K 以下の細胞では top-K Jaccard が常に 1 になり、偽の収束になる。
- **停止は細胞単位**。停止した細胞にも残りの摂動 step はかかるが、rerank はされない。全細胞が停止するか cap に達したら、それ以降の rerank 呼び出しを省く。cap 到達時にまだ動いていた細胞は `cap` として記録する。
- 細胞ごとの停止理由・停止 step・適用回数は条件ディレクトリの `feedback_guard.csv` に書く。条件ごとの集計は `run_manifest.json` の `multi_step.guard` に入る。feedback 行には `guard_active_before/after`, `guard_new_cycle_halts`, `guard_new_converged` も付く。
- 配線したのは**安全装置だけ**。multi-step が単発 feedback より有益か、安定かは評価していない（Phase 3）。

BBRC OSKM の 4-step（KLF4→MYC→SOX2→POU5F1）を 30 細胞で回した smoke では、`oracle` は step 2・3 の更新が小さく（ρ ≈ 0.998）、全細胞が step 3 で converged として停止した。`norm` と `linear_deltarank` は更新が大きいまま（ρ 0.95–0.99）3 回とも適用され、cycle は 0 だった。


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
| `core/state_feedback/controls.py` | base-rank control (partial ρ, cross-fitted base-only predictor) and perturbation-specificity helpers |
| `core/run_state_feedback_isp.py` | runner comparing all conditions in one run; optional perturbation-specificity stage |
| `core/config/state_feedback_isp.yaml` | config (OSKM values are a worked example only) |
| `tests/test_state_feedback_phase12.py`, `tests/test_state_feedback_controls.py` | with `test_state_feedback_oracle.py`: 81 tests, all pass in Docker |

**How to run** (host `spark-943a`, Docker):

```bash
ISP_PLATFORM_IMAGE=isp-platform:latest DOCKER_UID=$(id -u) DOCKER_GID=$(id -g) \
  docker compose run --rm --no-deps ordered_rank_edit_isp \
  python3 core/run_state_feedback_isp.py --config core/config/state_feedback_isp.yaml --max-ncells 50
# tests
ISP_PLATFORM_IMAGE=isp-platform:latest DOCKER_UID=$(id -u) DOCKER_GID=$(id -g) \
  docker compose run --rm --no-deps ordered_rank_edit_isp \
  python3 -m unittest tests.test_state_feedback_phase12 tests.test_state_feedback_controls tests.test_state_feedback_oracle
# perturbation specificity with an existing decoder (no retraining, no endpoint conditions)
  python3 core/run_state_feedback_isp.py --config ... --eval-only --specificity \
    --decoder-checkpoint <run>/decoder/delta_rank_decoder.pt
```

The specificity stage (`state_feedback.specificity`, off by default) writes
`<run>/perturbation_specificity/` (`perturbation_specificity.csv`, `vs_random.csv`,
`specificity_contrasts.csv`, `.json`). `direction_fidelity/` also gets
`base_rank_control.csv`, and `direction_fidelity.json` a `base_rank_control` block.

`--conditions` selects a subset (conditions are mutually independent, so they can be
split across jobs). Wall time at n=50 was ~26 min end to end; ~4 min per condition,
dominated by GPU forward passes and therefore roughly linear in cell count.

**Genericity.** The perturbation is whatever `state_feedback.steps` lists (any genes,
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

#### Phase 2 判定基準（二軸。endpoint shift 単独をゲートにしない）

Oracle-A は goal-state の観測 rank を次入力に入れるため endpoint 改善に leakage 構造を含む。feasibility / upper bound には必要だが **decoder の biological fidelity の評価ではない**。したがって:

| 軸 | Pass の目安 | 意味 | 現状 |
|---|---|---|---|
| **Direction fidelity**（主軸） | linear の held-out ρ が norm・ΔMLM を明確に上回る（bootstrap CI が 0 を跨がない）**かつ** base rank を除いた partial ρ の CI が 0 を跨がない | Δh から観測 rank displacement の**方向**を、元の位置だけで説明できる分を超えて読めている | **PASS**（partial ρ 0.24、下記） |
| Perturbation specificity | 評価対象の摂動の partial ρ がランダム摂動の分布を上回る | 上乗せが採点遺伝子の性質ではなく摂動の Δh に依存する | **PASS**（BBRC・Asano とも全ランダム組を上回る） |
| Effect prioritization | top-up/down の precision@K・NDCG@K が baseline を上回る | biologically relevant movers を優先できている | PASS（up 方向。down は弱い） |
| Calibration | Δr̂ の大きさと observed Δr の大きさが単調対応 | 順位変化量を過度に誇張・圧縮していない | 単調性 PASS、**magnitude は圧縮**（下記） |
| Null stability | null rerank が不変、または定義済み許容範囲内 | 無根拠な drift を導入しない | PASS（完全不変） |
| External validity | held-out perturbation / donor / dataset でも優位 | teacher 固有の shortcut でない | **一部検証**（別 dataset・別種: Asano PIPseq。donor / batch holdout は未） |
| Classifier consistency | endpoint shift が direction fidelity と少なくとも弱く正に相関 | downstream score が補助指標として整合 | **不整合**（下記） |

#### Direction fidelity 実測（n=50, held-out genes 2,189 / 25,047 samples, 同一教師・同一 split・同一単位）

| Method | pooled ρ | cell-wise median ρ (IQR) | gene-agg ρ | P@100 up | P@500 up | P@100 down | NDCG@100 up | calibration 単調性 |
|---|---|---|---|---|---|---|---|---|
| **`linear_deltarank`** | **+0.419** | **+0.443** (0.063) | **+0.613** | **0.230** | **0.646** | 0.100 | **0.407** | **+1.00** |
| `norm` | +0.028 | +0.025 (0.074) | +0.149 | 0.090 | 0.270 | 0.080 | 0.202 | +0.20 |
| `delta_mlm` | −0.016 | −0.002 (0.078) | −0.038 | 0.030 | 0.142 | 0.100 | 0.051 | −0.15 |
| `random` | −0.002 | −0.004 (0.045) | −0.020 | 0.050 | 0.166 | 0.070 | 0.084 | −0.07 |
| `identity` | 未定義 | — | — | 0.060 | 0.258 | 0.030 | 0.135 | −0.27 |

chance level は `K / n_genes` なので P@100 ≈ 0.046、P@500 ≈ 0.228。linear は P@100up で **5.0× chance**、P@500up で 2.8× chance。

**統計的不確実性**（cell-level bootstrap 1000 回、paired sign-flip permutation 10,000 回）:

| 対比 | Δρ | 95% CI | 0 を除外 | p | cell 単位で linear が勝った割合 |
|---|---|---|---|---|---|
| linear − `norm` | **+0.391** | [+0.365, +0.418] | **yes** | 1.0e-4 | **50/50** |
| linear − `delta_mlm` | **+0.435** | [+0.409, +0.462] | **yes** | 1.0e-4 | **50/50** |
| linear − `random` | +0.421 | [+0.398, +0.445] | yes | 1.0e-4 | 50/50 |

#### Phase 2 gate: **主軸 PASS**（Phase 3 は依然不要）

`norm` は pooled ρ = 0.028 で実質無情報。endpoint shift で decoder と同着（20.5% 対 20.8%）だったのは**指標側の縮退**であり、decoder 側の失敗ではなかったことが確定した。50 細胞すべてで linear が両 baseline を上回り、bootstrap CI は 0 を大きく離れている。

**それでも残る 4 つの限界を明記する。**

1. **Calibration は単調だが magnitude が圧縮されている。** 予測レンジ −0.073…+0.163 に対し観測レンジ −0.192…+0.096。左方向（up）の移動量を過小に、右方向を過大に見積もる。方向は正しく、量は信用できない。
2. **down 方向が弱い。** P@100 down = 0.100（2.2× chance）で ΔMLM と同値、P@50 down = 0.040（1.75× chance）。優位性は主に up-mover の同定に由来する。endpoint 教師の "down" 側は他遺伝子が左に動いた結果の相対的押し出し（compositional）で特異性が低いことと整合する。
3. **endpoint shift と direction fidelity が整合しない。** `delta_mlm` は endpoint shift 最高（gap closed 21.8%）でありながら direction fidelity は chance 以下（ρ = −0.016）。Classifier consistency 軸は満たされていない。endpoint shift は本問題では補助指標にもならない。
4. **external validity は一部のみ。** 下記「外部妥当性」「現 dataset で計算できないもの」参照。

#### Base-rank 対照（2026-09-27 追加。pooled ρ の読み方を改める）

endpoint 教師では、遺伝子が細胞内で元々どの位置にいたか（base rank）だけで観測 Δrank の多くが予測できる（上位の遺伝子は下がり、下位の遺伝子は上がる）。base rank だけを使う予測器（20 分位 bin の平均 Δrank、token の偶奇で 2 分割して交差適合 = `base_rank`）は、pooled ρ で linear とほぼ並ぶ。

| n=300, BBRC | pooled ρ | partial ρ given base rank [95% CI] |
|---|---|---|
| `linear_deltarank` | 0.437 | **0.242** [0.234, 0.250] |
| `base_rank`（交差適合） | 0.422 | 0 以下（位置以外の情報を持たない。5 細胞 smoke で −0.06） |
| linear、Δh を細胞内でシャッフル | — | 0.010 |
| linear、base を 0.5 に固定（Δh のみ） | — | ≈ linear と同値 |
| `norm` / `delta_mlm` | 0.040 / −0.013 | — |

- partial ρ = base rank の 20 分位 bin ごとに、予測と観測の順位からそれぞれ bin 平均を引いた後の相関。「元の位置を超えて Δh が足している分」
- **上の実測表の pooled ρ 0.419 の大部分は base rank で説明される。** decoder の Δh 由来の上乗せは partial ρ ≈ 0.24。判定基準にこれを加えた（`direction_fidelity_verdict` の `adds_beyond_base_rank`）
- Δh をシャッフルすると上乗せは消え、base を固定しても残る。上乗せは Δh の向きに由来する

#### 摂動特異性（同じ decoder に別の摂動の Δh を入れる）

decoder（OSKM の Δh で学習）・teacher・held-out gene・細胞は固定し、Δh を作る摂動だけを変える。上乗せが採点遺伝子の性質だけなら、摂動によらず同じ値になるはず。摂動した遺伝子は全条件で採点から除外。

| BBRC | partial ρ n=50 [95% CI] | partial ρ n=300 [95% CI] |
|---|---|---|
| OSKM（config の chain） | 0.229 [0.210, 0.248] | 0.241 [0.234, 0.250] |
| OSKM 同時 OE | 0.228 [0.208, 0.246] | 0.240 [0.233, 0.249] |
| 3F（POU5F1, SOX2, NANOG） | 0.249 [0.230, 0.269] | 0.264 [0.256, 0.273] |
| 7F（NANOG, POU5F1, SOX2, ESRRB, LIN28A, DPPA4, TERT） | 0.289 [0.274, 0.306] | 0.307 [0.300, 0.315] |
| ランダム OE 4 genes（n=50: 30 組, n=300: 10 組）平均 ± SD（最大） | 0.075 ± 0.026（0.156） | 0.075 ± 0.028（0.128） |

- 4 条件とも全ランダム組を上回る（経験的 p は組数で決まる最小値: n=50 で 0.032、n=300 で 0.091）
- 7F > 3F > OSKM（n=300: 7F − OSKM = +0.067 [+0.063, +0.071]、3F − OSKM = +0.024 [+0.021, +0.027]）。これは「model 空間で somatic → pluripotent の endpoint 方向により揃う」という意味に限る。decoder は OSKM で学習し 3F・7F と因子を共有するため、reprogramming 効率の比較ではない
- OSKM の chain と同時 OE の差は +0.001 [+0.000, +0.002]。この指標では順序の効果は見えない
- ランダム OE の partial が 0 でない（0.075）のは、どの OE でも一部は endpoint 方向に揃うため

#### 外部妥当性: Asano PIPseq（mouse 大動脈 scRNA-seq, AD → WT）

mouse Geneformer の FT モデル（AD/WT）と PIPseq のみの tokenized dataset。摂動は Igfbp2 delete、observed_state = WT。hidden size が違うため decoder は Asano 上で学習し直した（他のハイパーパラメータは BBRC と同一）。

| Asano | n=50 | n=300 |
|---|---|---|
| linear pooled ρ / base-only（交差適合） | 0.229 / 0.159 | 0.235 / 0.171 |
| linear partial ρ [95% CI] | 0.133 [0.110, 0.155] | 0.125 [0.116, 0.135] |
| Δρ vs norm | +0.104 [+0.075, +0.132] | +0.114 [+0.103, +0.125] |
| ランダム delete 1 gene（Igfbp2 と検出頻度 0.5〜2 倍）partial 平均（最大） | 0.006（0.027）, 30 組 | 0.007（0.028）, 10 組 |

- 別の種・別の遷移・別の FT モデルでも「norm・ΔMLM を上回り、base rank を超える上乗せがあり、ランダム摂動を上回る」形は再現した。大きさに依存しない Δh のみの partial でも Igfbp2 はランダムと分離する
- 効果は BBRC の約半分。AD と WT の差自体が小さく、混合細胞集団の pseudobulk なので細胞組成の差も teacher に混じりうる
- ランダム delete は検出頻度では揃えたが、細胞内の順位は揃えていない。decoder は Igfbp2 の Δh で学習している
- PIPseq のみなので batch holdout はしていない

#### 現 dataset で計算できないもの（黙って省略せず明記する）

| 解析 | 不可の理由 |
|---|---|
| within-gene / across-cells Spearman | pseudobulk teacher は**遺伝子ごとに定数**なので細胞方向の分散がゼロ。定義不能。gene 単位の一致は gene-aggregated ρ で測り、gene identity 記憶は held-out gene split 自体が対照になる |
| perturbation-wise（観測側） | 摂動ごとの観測 post-perturbation rank が無い。decoder 側の摂動特異性は上記の対照で見ている |
| donor / batch 層別 | `replicate` は `Rep1` 単一、`sample_id` は `state_key` と完全共線 |
| Oracle-B (matched perturbation) | OSKM 誘導後の観測中間時点が無い |
| cell-state-pair / dataset holdout | 別 transition / 別 dataset の FT モデルが未整備 |

**batch holdout 用の候補資産:** Asano mouse dataset は `3w`/`5w` × `WT`/`AD` × replicate `1st`/`2nd`/`PIPseq` を持つ。上の外部妥当性は PIPseq のみの FT モデルで行った。batch holdout には replicate をまたぐ FT と評価が別途必要。

#### 現時点で証明できていること / いないこと

**証明できたこと:**

- Oracle-A reranking は endpoint classifier 空間を大きく動かせる（feasibility）
- `null_feedback` は完全な恒等置換であり、decoder は無根拠な drift を作らない（変位 0.0 / Spearman 1.0 / top-100 Jaccard 1.0 / shift 0.0000）。解析 null drift は非ゼロ（0.004）だが `base_rank_norm` の単調関数なので順序不変
- linear decoder は held-out gene 上で観測 Δrank に対し pooled ρ = 0.419、cell-wise median ρ = 0.443 を示し、`norm` / `delta_mlm` を bootstrap CI が 0 を跨がない差で上回る（50/50 細胞で優位）。ただし pooled ρ の大部分は base rank で説明でき、Δh 由来の上乗せは partial ρ ≈ 0.24（n=300 で CI [0.234, 0.250]）
- その上乗せは摂動に依存する（初期化カクテル 0.24〜0.31 対 ランダム OE 0.075）
- 同じ手順で別の種・別の遷移（Asano PIPseq）でも、小さいながら同じ形が再現する（partial ρ ≈ 0.13、ランダム delete ≈ 0.01）
- endpoint shift は direction-sensitive な decoder 評価として**不十分**（`delta_mlm` が endpoint 最高かつ direction chance 以下）

**まだ証明できていないこと:**

- decoder を学習に使っていない摂動へ持ち込んだときに通用するか（現在はどちらの dataset でも評価対象の摂動で学習）
- donor / batch をまたいで一般化するか
- 変位の **大きさ** が信用できるか（単調だが圧縮されている）
- down 方向（発現順位低下）を特異的に捉えられているか
- multi-step state-feedback が単発 feedback を越えて有益かつ安定か（cycle detection / convergence stopping は配線済み、評価は未実施）

#### 次にやること（優先順）

1. `max_ncells: 300` で Phase 0 Oracle-A confirmation を再測（direction fidelity と摂動特異性は n=300 で再測済み、結論は n=50 と同じ）
2. `max_shift` grid をさらに上へ（val Spearman は 0.5 で `0.458` と**まだ伸びている**。ただし 0.5 は encoding の半分を 1 step で動かせる幅なので、生物学的妥当性の上限も併せて決める）
3. calibration の magnitude 圧縮を是正（Huber 重み / 出力スケールの再検討）
4. down 方向の弱さの原因究明（compositional な押し出しか、教師の非対称性か）
5. Oracle-B 用の reprogramming time-course dataset を用意
6. Asano: batch holdout（replicate をまたぐ FT）と、in-cell rank も揃えたランダム delete
7. multi-step の有益性・安定性の評価（cycle detection / convergence stopping / hard cap は runner に配線済み。下記 stability metrics を multi-seed で報告）
8. **Phase 3 (MLP) は上記が済み、かつ linear が Oracle-B gap を残した場合のみ**


## 必須比較条件

| Condition | Description |
| --- | --- |
| Ordered rank-edit (current) | Paper baseline; token-edit chain, no embedding feedback |
| No-feedback | 1st step rank list frozen for all subsequent steps |
| Identity (Δr̂ = 0) | 厳密 null。定数予測なので Spearman は定義されない（それが正しい答え） |
| Random bounded shift | rank-independent noise。permutation null |
| Norm reranking | Parameter-free unsupervised scalar baseline（勝てなければ decoder の主張が成立しない） |
| ΔMLM + inertia | Pretrained-head differential baseline |
| Base-rank only | 20 分位 bin の平均 Δrank（交差適合）。endpoint 教師で位置だけから取れる分の対照。主法はこれを partial ρ で超える必要がある |
| Linear residual Δrank | **主法** |
| Linear, other perturbations | 同じ decoder に別の摂動・ランダム摂動の Δh。上乗せの摂動特異性の対照 |
| MLP residual Δrank | Phase 3 のみ。上限比較 |
| **Oracle-A (endpoint rank)** | 理論上限 / feasibility。**endpoint leakage を含むため fidelity 目標ではない** |
| **Oracle-B (matched perturbation)** | 同一摂動の実測 post-perturbation rank。leakage なしの fidelity 目標（データ待ち） |


## Multi-step stability metrics (T ≥ 2 で報告)

- Per-step rank correlation: Spearman(X_t, X_{t+1})
- Top-K overlap (Jaccard) at K = 100, 500, 1000
- Rank displacement distribution per step
- Null-perturbation cumulative drift
- 2-cycle / oscillation frequency（`feedback_guard.csv` の `halt_reason = cycle`）
- Multi-seed reproducibility
- Convergence step count distribution（`feedback_guard.csv` の `halt_step`）


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
  + λ₂ · L_pair           (pairwise / ListMLE ordering loss) ※Δr に対して掛ける
  + λ₃ · L_identity        (ctrl→ctrl: Δr ≈ 0)
  + λ₄ · L_smooth          (regularize extreme Δr)
```

`L_pair` は **Δr（変位）の順序**に対して定義する。絶対 priority (`base_rank_norm + Δr`) の順序に対して定義してはならない — `base_rank_norm` を tanh で増幅すれば満たせてしまい、Δh 由来の情報が失われる（上記 decoder 節参照）。

実装既定値 (`state_feedback.decoder`): `lam_huber: 1.0`, `lam_pair: 0.5`, `lam_identity: 1.0`, `lam_smooth: 0.01`, Huber δ = 0.05, Adam。


## scPRINT-2 reference (design parallel, not drop-in)

Found under `~/20260323scPRINT-2/` (also `~/20260531scPRINT2andPanOrganseq/scPRINT-2-private/`):

- Spec: `docs/ISP_v2_spec.md`
- `predict_expression` → `pred_start.h5ad` / `pred_pert.h5ad` / `delta_ranked_genes.csv`
- Enrichment: `services/run_isp_enrichment.py` (gseapy prerank)

scPRINT predicts expression via ZINB mean (`scprint_mu`), then ranks deltas. This is analogous to MLP Δrank + expression decoder — natural for scPRINT's architecture but inapplicable to Geneformer (no expression decoder). The residual Δrank approach above achieves a comparable pipeline shape without requiring an expression decoder.
