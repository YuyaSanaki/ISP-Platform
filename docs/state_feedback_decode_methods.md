# State-feedback ISP v1.0.1 — permutation decoder 設計

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

**Key constraint**: 遺伝子集合は固定。Token の挿入/削除を許すと generative transcriptome completion になり、別の問題。State-feedback ISP v1.0.1 は **fixed detected-gene set 上の conditional reranking**。


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
| norm ≈ decoder（direction fidelity でも） | `norm` は null ではない。v1.0.1 の失敗ではなく、hidden-state norm に position / rank / context の実在シグナルが漏れていることを意味する。その場合 “negative control” の呼称をやめ **nonparametric representation baseline** と呼び替える |

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
        self.proj = nn.Linear(d_model, 1, bias=False)   # Δh のみ、bias なし
        self.max_shift = max_shift

    def forward(self, h_pert, h_ctrl, base_rank_norm):
        delta_h = h_pert - h_ctrl          # 摂動効果のみ
        delta_rank = self.max_shift * torch.tanh(self.proj(delta_h).squeeze(-1))
        return base_rank_norm + delta_rank  # argsort → new input_ids
```

**入力設計の意図**: `delta_h`（`h_ctrl` ではなく差分）を使うことで、decoder が base cell identity ではなく **perturbation effect** を読むよう制約。`h_ctrl` の追加は後続 ablation。base rank と bias も入力に入れない。これにより Δh = 0 なら学習した重みによらず変位は厳密に 0 になり、遺伝子の移動が元の位置だけで決まることもない。当初は `[delta_h ; base_rank_norm]` と bias を入力にしていたが、base を 0.5 に固定した Δh のみの予測でも partial ρ が linear と同等だった（BBRC n=300: 0.332 対 0.329）ため外した。旧形式で保存した decoder は読み込めない（再学習が必要）。

**`tanh` bound の意図**: 1 step で bottom → top のような病的ランクジャンプを抑止。`max_shift` は decoder の出力（1 回の feedback で 1 遺伝子が動ける幅、細胞内 encoding 長に対する割合）の上限で、Geneformer 自体の出力は抑えていない。grid search（実装では val Spearman が 0.20 で伸び続けたため {0.05, 0.1, 0.2, 0.3, 0.5} に拡張）。

**実装との対応** (`core/state_feedback/decoder.py`): 上記 `forward(h_pert, h_ctrl, base_rank_norm)` は `from_states()` として保持し、中核は `delta_h` を直接受ける `delta_rank(delta_h)` / `forward(delta_h, base_rank_norm)`。同じ `delta_h` を学習と推論で再利用するため。`proj` は **zero-init** なので未学習 decoder は厳密に恒等置換になる。

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
| 3 | **Null-drift check** | null perturbation → Δr ≈ 0 を検証、drift 量を報告 | 済 — decoder の入力が Δh のみ（bias なし）なので Δh = 0 で変位は構造的に 0。解析値 `decoder.null_drift`（常に 0）と実走条件 `null_feedback` で毎回記録する。旧 decoder（base rank + bias 入り）でも実測は**変位 0.0 / Spearman 1.0 / shift 0.0000**だった（解析 drift 0.004 は `base_rank_norm` の単調関数で順序不変だったため。ただし構造的な保証ではなかった） |
| 4 | **Swap hysteresis** | \|ŝ_i − ŝ_j\| < ε の隣接ランクは元の順序を保持 | 済 — priority を ε グリッドに量子化してから**安定ソート**（近接差を厳密な同値に変える）。既定 `hysteresis: 0.0`（無効） |
| 5 | **Cycle detection** | X_t ↔ X_{t+1} の 2-cycle を検出、halt | 削除（下記） |
| 6 | **Convergence stopping** | Spearman(r_t, r_{t+1}) > 0.995 × 2 step、または top-1000 Jaccard > 0.99、または hard cap T ≤ 5 | 削除 — feedback 回数は常に step 数（下記）。収束停止・cap とも削除 |
| 7 | **Cell-wise normalization** | ランクは cell-specific 相対量; 異なる sequence length の細胞間でスコアを直接比較しない | 済 — `base_rank_norm = i/(n−1)` で細胞内正規化、z-score も細胞内。教師 Δrank も各 state の平均 encoding 長で正規化 |

feedback は全 step の後（最終 step の後を含む）にかける。これ以外のスケジュールと回数の上限はない（2026-09-30 に削除）。feedback 回数は常に step 数と同じで、終点 shift は最後の rerank 後の encoding で読む。step 間に feedback がなければ結果は最終 encoding だけで決まり、Ordered rank-edit ISP（= 逆順リストの multi-gene ISP）と同じになる。最終 step 後だけの feedback も最終 encoding だけで決まる。したがって逐次（sequential）と言えるのは毎 step feedback だけである。旧キー `feedback_every_step` / `feedback_after_step` / `feedback_after_last_step` / `multi_step` は、毎 step feedback 以外を指定するとエラーになる（`state_feedback.multistep.check_feedback_config`）。`pin_overexpressed: true`（既定）では、それまでに OE した遺伝子を rerank 前の順序のまま先頭に固定し、残りの遺伝子だけを並べ替える。各 feedback 行には `feedback_event`（= step 番号）が入る。

- **細胞単位の停止は置かない**。
  - 収束停止（削除）: 各 step で新しい摂動が入るので、これまでの feedback の変化が小さくても次の step の feedback が小さいとは限らない。encoding 全体の Spearman / top-K Jaccard の閾値は少数遺伝子の大きな移動も見逃す（2048 遺伝子の細胞で 1 遺伝子が最下位→最上位に動いても Spearman 0.997、4096 遺伝子では 0.9985）。当初は 0.995 × 2 回連続で停止していたが、マスターレギュレーター的な少数遺伝子の変化を「収束」とみなして以降の feedback を止めるため削除した。
  - 2-cycle 停止（削除）: 順位リスト全体が 2 回前と完全一致したときだけ発動するが、feedback の間に新しい摂動が入るので実際にはほぼ発動しない（下記 smoke でも 0 件）。
  - 旧設定の `converge_*` / `halt_on_cycle` キーは警告を出して無視する。`feedback_guard.csv` は書かない。

> **注意：multi-step feedback は回を重ねるごとに計算上の誤差が膨らむ。** decoder は start→全 step の Δh から観測された終点の順位変化全体を予測するよう学習されている。毎 step feedback をかけると（`ctrl_reference: start`）、毎回 start 基準で Δh を取り直し（それまでの step と rerank の効果を含む）、すでに動いた順位にさらに終点規模の変化を足す。変化量と decoder の誤差は少なくとも回数に比例して増え、観測された終点を超えた順位では shift はモデルが持つ本来の生物学的意味ではなく encoding の乱れを反映し、生物学的な信号がマスクされていく。`ctrl_reference: previous` はそれまでの step の二重計上を避けるが、1 回ごとに終点規模の変化を予測する点は変わらない。1 回を超えて信号が誤差に埋もれないと予測できる回数は見つかっていない。**step 数（= feedback 回数）が多いほど誤差は大きい。** chain は短く保ち、step 数の同じ chain どうしだけを比べ、生物学的な主張の前には対応するランダム chain の gain を差し引く（下記「Multi-step stability evaluation」「ランダム chain の設計」）。BBRC OSKM ではランダム分が大きく（gain の約 94%）、今後の改善が必要である。

BBRC OSKM の 4-step（KLF4→MYC→SOX2→POU5F1）を 30 細胞で回した smoke（細胞単位の停止を削除する前、旧 decoder）では、`oracle` は step 2・3 の更新が小さく（ρ ≈ 0.998）、全細胞が step 3 で converged として停止した。`norm` と `linear_deltarank` は更新が大きいまま（ρ 0.95–0.99）3 回とも適用され、cycle は 0 だった。目標そのものである `oracle` が 1 回でほぼ止まる一方、decoder は動き続けており、上の注意と整合する。


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

### Phase 0 data

- Dataset: **BBRCv3** (or later), copied to `data/bbrcv3/tokenize_human_native` and `data/bbrcv3/v2_ft_human_native` (`/data/` is gitignored)
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

**How to run** (Docker):

```bash
DOCKER_UID=$(id -u) DOCKER_GID=$(id -g) \
  docker compose run --rm --no-deps ordered_rank_edit_isp \
  python3 core/run_state_feedback_isp.py --config core/config/state_feedback_isp.yaml --max-ncells 50
# tests
DOCKER_UID=$(id -u) DOCKER_GID=$(id -g) \
  docker compose run --rm --no-deps ordered_rank_edit_isp \
  python3 -m unittest tests.test_state_feedback_phase12 tests.test_state_feedback_controls tests.test_state_feedback_oracle
# perturbation specificity with an existing decoder (no retraining, no endpoint conditions)
  python3 core/run_state_feedback_isp.py --config ... --eval-only --specificity \
    --decoder-checkpoint <run>/decoder/delta_rank_decoder.pt
```

`--decoder-checkpoint` は同じ step リストで学習した decoder の再利用（条件や specificity の追加）向け。別の step リストに使うのは転移テストで、同じ遺伝子の順番違いも含む（最後に OE した遺伝子が encoding の先頭に来るので Δh が変わる）。BBRC OSKM（n=3000、24 order、毎 step feedback）で、同時 OSKM で学習した decoder を全 order に使い回すと、order ごとに学習した decoder と一致したのは POU5F1 が最後の 6 order（先頭の並びが同時 OE と同じ）だけだった。残り 18 order では direction fidelity（pooled Spearman）が 0.32〜0.36 に落ち、order ごとの decoder は 0.42〜0.47（24 order 平均 0.38 vs 0.46、base rank を除いた partial ρ 0.29 vs 0.35）。終点の order 順位は両者でほぼ無相関（Spearman −0.02）。報告する結果には run ごとに学習した decoder を使う（Web UI の既定）。

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
| linear、base を 0.5 に固定（Δh のみ） | — | ≈ linear と同値（この結果を受けて decoder を Δh のみの入力に変更。以下の数値は変更前の decoder） |
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
- `null_feedback` は完全な恒等置換であり、decoder は無根拠な drift を作らない（変位 0.0 / Spearman 1.0 / top-100 Jaccard 1.0 / shift 0.0000）。旧 decoder では解析 null drift が非ゼロ（0.004）だったが `base_rank_norm` の単調関数なので順序不変。現在の decoder（Δh のみ、bias なし）では構造的に 0
- linear decoder は held-out gene 上で観測 Δrank に対し pooled ρ = 0.419、cell-wise median ρ = 0.443 を示し、`norm` / `delta_mlm` を bootstrap CI が 0 を跨がない差で上回る（50/50 細胞で優位）。ただし pooled ρ の大部分は base rank で説明でき、Δh 由来の上乗せは partial ρ ≈ 0.24（n=300 で CI [0.234, 0.250]）
- その上乗せは摂動に依存する（初期化カクテル 0.24〜0.31 対 ランダム OE 0.075）
- 同じ手順で別の種・別の遷移（Asano PIPseq）でも、小さいながら同じ形が再現する（partial ρ ≈ 0.13、ランダム delete ≈ 0.01）
- endpoint shift は direction-sensitive な decoder 評価として**不十分**（`delta_mlm` が endpoint 最高かつ direction chance 以下）

**まだ証明できていないこと:**

- decoder を学習に使っていない摂動へ持ち込んだときに通用するか（現在はどちらの dataset でも評価対象の摂動で学習）
- donor / batch をまたいで一般化するか
- 変位の **大きさ** が信用できるか（単調だが圧縮されている）
- down 方向（発現順位低下）を特異的に捉えられているか
- multi-step state-feedback が単発 feedback を越えて有益かつ安定か（hard cap のみ配線済み、評価は未実施。回数とともに誤差が膨らむ構造的な理由は「実装ガードレール」節の注意を参照）
- Δh のみの decoder で上記の数値（旧 decoder: Δh + base rank + bias）が再現するか

#### 次にやること（優先順）

1. `max_ncells: 300` で Phase 0 Oracle-A confirmation を再測（direction fidelity と摂動特異性は n=300 で再測済み、結論は n=50 と同じ）
2. `max_shift` grid をさらに上へ（val Spearman は 0.5 で `0.458` と**まだ伸びている**。ただし 0.5 は encoding の半分を 1 step で動かせる幅なので、生物学的妥当性の上限も併せて決める）
3. calibration の magnitude 圧縮を是正（Huber 重み / 出力スケールの再検討）
4. down 方向の弱さの原因究明（compositional な押し出しか、教師の非対称性か）
5. Oracle-B 用の reprogramming time-course dataset を用意
6. Asano: batch holdout（replicate をまたぐ FT）と、in-cell rank も揃えたランダム delete
7. multi-step の有益性・安定性の評価（hard cap のみ runner に配線済み。下記 stability metrics を multi-seed で報告）
8. **Phase 3 (MLP) は上記が済み、かつ linear が Oracle-B gap を残した場合のみ**
9. **逐次フィードバックの誤差蓄積（将来）。** 毎回 feedback は自分の出力を次の入力に戻す閉ループで、
   S1（event ごとの変位の増大）と S2（新しい編集が無くても rerank が収束しない）はこの誤差蓄積の症状。
   他分野の対処: 自分のロールアウトで集めた状態で再学習する（DAgger, Ross et al. 2011; scheduled sampling,
   Bengio et al. 2015）、複数ステップを展開した損失で学習する（unrolled training / pushforward trick,
   Brandstetter et al. 2022）。ここでは、feedback 後の encoding から Δh を集めて同じ教師で decoder を
   再学習する、T step を展開して終点の教師との誤差で学習する、が候補。S2 の一因である `ctrl_reference: start`
   による変位の二重計上の扱いも併せて決める。評価は S1/S2 と idle rerank の収束で行う。
10. **遺伝子に依らない gain の分離（Plan B / 将来）。** マッチしたランダム chain でも gain の 95–97% が出る
   （上記 X5-R 結果）。decoder は教師が一つ（somatic → pluripotent）なので、Δh に含まれる遺伝子の識別情報から
   教師を引き当てられる可能性がある。報告の軸はプラセボとの差（Δspec, 計量経済のプラセボ検定・摂動予測の
   Systema と同じ考え方）とし、将来の改善として decoder の入力側で遺伝子に依らない成分を除く:
   (a) 概念消去（プラセボで当てはめた回帰で、遺伝子の文脈表現 h_ctrl から予測できる Δh 成分を除く;
   base rank は LEACE, Belrose et al. 2023 / INLP, Ravfogel et al. 2020）、(b) 潜在反応の考え方での
   プラセボ対照 Δh（細胞・遺伝子ごとに Δh − プラセボ chain の平均 Δh）。評価は decoder の selectivity
   （OSKM とランダムの val Spearman の差、Hewitt & Liang 2019 の control task）と Δspec の特異的な割合。
   少数細胞のスクリーニングは `review/REVISION_TODO.md` X7。
   **方針（2026-10-01 決定）: 潜在反応（ループ内）と Δspec（結果側）を組み合わせる。**
   - ループ内（V2, 潜在反応のプラセボ対照）: decoder は Δh − (同じ細胞・同じ遺伝子の推定用プラセボ chain の
     平均 Δh) で学習する。各 rerank では Δh − m_g を入れる。m_g は、現在のエンコーディングで chain が
     挿入した遺伝子を推定用プラセボの遺伝子に置き換えたときの Δh の平均（同じ状態で遺伝子だけが違う
     反実仮想の応答）。
   - 結果側（Δspec, 細胞ごと）: τ_i = gain_i(因子) − mean_p gain_i(p)。gain は Ordered rank-edit からの
     上乗せ、p は評価用プラセボ chain（推定用とは別抽出; プラセボ自身の評価は leave-one-out）。
   - 少数細胞スクリーニング（`x7_causal.py` / `analyze_x7c_methods.py`, OSKM K→M→S→O, 50 cells,
     評価用プラセボ 50 本, decoder seed 0/1/2）:

     | 方法 | OSKM 推定値 | z | 51 本中の順位 | 全プラセボを上回る細胞 | 陽性対照 OSK z / OSNL z |
     | --- | --- | --- | --- | --- | --- |
     | Δspec のみ（V0） | 0.0026 / 0.0058 / 0.0073 | 0.6–1.4 | 14 / 10 / 5 | 0–12% | −3.9〜−5.7 / 4.2 |
     | 潜在反応のみ（V2, 生値） | 0.226 | 6.3 | 1 | 100% | 4.5 / 11.5 |
     | 潜在反応 + Δspec | 0.059 / 0.055 / 0.068 | 5.5–6.7 | 1（並べ替え p = 1/51, 下限） | 100% | 3.9–5.0 / 11.3 |

     潜在反応のみではプラセボも OSKM 推定値の 73% を得る（遺伝子に依らない成分が残る）ため、結果側の
     Δspec と組み合わせる。組み合わせではプラセボ平均は定義上 0、最大のプラセボでも OSKM の 53–63%。
     Δspec のみでは OSKM がプラセボと区別できず、陽性対照 OSK が負になる。媒介分析: 遺伝子に依らない
     媒介 m_g だけで V0 の OSKM gain の 99% が再現される。
   - 前日の検討（概念消去 V1）: decoder 単体の selectivity は約 2 倍になったが、ループ内の特異的な割合は
     1–2 ポイントの増加に留まった（OSKM 2.0% → 3.0%、7F 7 ステップ 9.0% → 10.2%、7F 1 ステップ
     23.2% → 25.5%; `x7_7f.py`）。消去なしの Δspec でも 7F は全 run で 1 位（0.0275, z 6.0）、OSKM は
     0.0058（z 1.0）。診断: プラセボ平均 Δh が Δh 分散の 87–96% を説明し、教師は h_ctrl だけから
     Spearman 0.79 で予測できる。
   - 残り: 本文に引用する前に、最終コードで 300 cells・seed 0/1/2 の再測定を行う。


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
- Multi-seed reproducibility

### Multi-step stability evaluation（事前登録 2026-09-29、実行前に固定）

Runner: `core/run_state_feedback_stability.py`（設定は `state_feedback.stability`）。
各 decoder seed（既定 0, 1, 2）で decoder を学習し直す。train/val の遺伝子 split は
`decoder.seed` で固定する。すべて cell-mean cosine の goal-state shift と、
両 encoding に共通する token 上の順位比較で測る（OE で末尾 token が落ちても定義できる）。

| 系列 | 内容 |
|---|---|
| Ordered rank-edit | decoder なし。各 chain の参照経路（seed に依存しないので 1 回） |
| configured, multi-step | 毎 step feedback（当時の `feedback_every_step: true`）, 最終 step 後も feedback, OE pin, `ctrl_reference: start` |
| idle events | configured chain の終点から、新しい摂動なしで `idle_events` 回（既定 5）rerank を繰り返す。cap を超えるのは診断のため |
| configured, single event | `feedback_every_step: false`（step 1 後の 1 回のみ）。比較用 |
| random chains | configured と同じ step 数・同じ型で、teacher 遺伝子から 1 step 1 遺伝子（既定 5 本、`random_seed` で固定し全 seed で同一） |

判定（configured chain について。S1–S3 は全 seed で成立すること）:

| ID | 問い | 基準 |
|---|---|---|
| S1 | 1 回の feedback の変化は有界で、回を追って膨らまないか | feedback 各回の per-event Spearman（細胞中央値）が 1 回目から 0.02 超下がらない。かつ per-event 変位（細胞中央値の中央値, 位置単位）の最大が 1 回目の 1.5 倍以下（1 回目の下限は 1 位置） |
| S2 | 摂動が止まれば encoding は収束するか（drift しないか） | 最後の idle event の per-event Spearman ≥ 0.99、かつ最後の idle 変位 ≤ 最初の idle 変位 |
| S3 | 終点の上乗せは摂動特異的か（endpoint 教師への一律の引き寄せでないか） | configured の endpoint gain（multi-step − Ordered rank-edit）がどの random chain の gain よりも大きい |
| S4 | decoder seed で結果が変わらないか | seed 間の最終 encoding の per-cell Spearman（細胞中央値）の最小 ≥ 0.95、かつ endpoint shift の seed 間 CV ≤ 10% |
| S5 | single と multi の差 | 報告のみ（基準なし）。endpoint、Ordered rank-edit からの距離 |

**決定規則:** S1–S4 がすべて成立したときに限り、multi-step を論文で結果として報告する。
一つでも不成立なら論文の State-feedback は single event（`feedback_every_step: false`）で
報告し、multi-step は不安定だった旨と数値を補遺に置く。基準は結果を見て変えない。

懸念とそれを測る系列: `ctrl_reference: start` では rerank 済みの変位も Δh に含まれるので、
feedback を重ねると同じ変位を二重に数えうる（idle events が測る）。decoder は endpoint
（観測 goal state）の教師で学習しているので、どの摂動でも goal へ引き寄せうる（random
chains が測る）。null（摂動なし）の multi-step は Δh = 0 で構造的に恒等なので測らない。

#### 結果（2026-09-29, Pegasus H100, somatic 300 cells, OSKM を K→M→S→O の順, seed 0/1/2）

**判定: S1 不成立, S2 不成立, S3 成立（差は小さい）, S4 不成立 → multi-step は支持されない。**
決定規則により、論文の State-feedback は single event（`feedback_every_step: false`）で報告する。
出力: `ISP-Platform/output/sf_stability_20260929/pegasus_out/aggregate/`（seed ごとの run は同階層）。

| 項目 | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| feedback 各回の Spearman（細胞中央値） | 0.974 / 0.811 / 0.877 / 0.923 | 0.982 / 0.838 / 0.856 / 0.898 | 0.974 / 0.826 / 0.878 / 0.912 |
| feedback 各回の変位（位置, 細胞中央値） | 78 / 258 / 198 / 144 | 65 / 239 / 218 / 179 | 78 / 247 / 204 / 156 |
| idle 1→5 の Spearman | 0.947 → 0.981 | 0.931 → 0.984 | 0.931 → 0.968 |
| idle 1→5 の変位 | 104 → 55 | 137 → 55 | 117 → 62 |
| idle 5 後の Ordered rank-edit 終点との Spearman | 0.158 | −0.077 | 0.056 |
| 終点 shift: multi-step（idle 前） | 0.308 | 0.329 | 0.342 |
| 終点 shift: idle 5 後 | 0.365 | 0.361 | 0.383 |
| 終点 shift: single event | 0.212 | 0.202 | 0.212 |
| random chain 5 本の終点 shift | 0.279–0.283 | 0.300–0.306 | 0.305–0.310 |

Ordered rank-edit の終点は OSKM 0.0105、random chain 0.0004–0.0029。seed 間の最終 encoding の
per-cell Spearman は 0.80 / 0.84 / 0.87（top-100 Jaccard 0.39–0.52）、終点 shift の CV は 5.2%。

読み方:

- **S1**: 2 回目の feedback で変化が 1 回目の約 3 倍（変位 65–78 → 239–258 位）に膨らむ。以降は縮むが 1 回目より大きい
- **S2**: 摂動なしの rerank でも 1 回あたり 55–62 位動き続け、終点 shift も上がり続ける。変位は減っているので発散ではないが、
  5 回では収束しない。その間に Ordered rank-edit の終点との相関は 0 付近まで落ち、元の encoding の順序はほぼ残らない。
  `ctrl_reference: start` の Δh が既に適用した変位を含むことによる二重計上と整合する
- **S3**: 基準上は成立するが、ランダム遺伝子 4 つの chain でも OSKM の gain の 91–95% が出る。multi-step の終点の上乗せの
  大部分は摂動に依らない goal 方向への引き寄せで、OSKM 特異的な部分は 0.015–0.022 程度
- **S4**: 終点 shift の値は seed 間で揃う（CV 5%）が、どの遺伝子がどこに来るかは seed で変わる（Spearman 0.80–0.87）
- single event の終点（0.20–0.21）も、1 回目の feedback だけで ~0.19 に跳ぶ。この跳びが OSKM 特異的かは単発でのランダム対照が要る
  （ここでは測っていない。摂動特異性の partial ρ は「摂動特異性」節）

上の single event は step 1 の後の 1 回（`feedback_every_step: false` の意味）。

### 毎回 feedback と単発 feedback の比較（事前登録 2026-09-30、実行前に固定）

完全な安定性は原理的に無い（rerank 後の encoding は start と異なるので Δh はゼロに戻らない）。
そこで S1/S2 の数値は不安定さの幅として報告し、主結果を毎回 feedback と単発 feedback の
どちらにするかは「OSKM に特異的な上乗せが両者でどれだけ違うか」で決める。上の S1–S4 の
判定（S1, S2, S4 不成立）はこの変更とともに併記する。

単発は **最後の step の後に 1 回**（`stability.single_event_at: last`）。同じ最終編集なら経路に
よらず同じ結果になる（`output/sf_path_dependence_20260929`）。ランダム chain 5 本（上と同じ遺伝子）と
seed 0/1/2（上と同じ decoder seed）で、毎回 feedback の既存 run と対にする。

指標（seed ごと、細胞で対にする）:

- 細胞ごとの gain = その chain の終点 shift − 同じ chain の Ordered rank-edit 終点 shift
- 特異的上乗せ Δspec = OSKM の gain − ランダム chain 5 本の gain の平均。細胞平均と、細胞 bootstrap
  （2,000 回）の 95% CI
- 特異的な割合 = Δspec / OSKM の gain（細胞平均どうし）

判定:

| ID | 基準 |
|---|---|
| D1 | 毎回・単発の両方で、全 seed の Δspec の CI が 0 を含まない |
| D2 | 全 seed で Δspec(毎回) / Δspec(単発) が 0.5〜2 |

**決定規則:** D1 と D2 が成立 → 毎回 feedback を主結果にする（Δspec、feedback 回数を揃えた比較のみ、
S1/S2 の幅を併記）。D1 が単発でのみ成立、または D2 不成立 → 単発 feedback を主結果にする。
単発でも D1 が不成立 → どちらの設定でも終点の上乗せは摂動特異的と言えないので、State-feedback の
終点 shift は主張に使わない。seed 間の最終 encoding の一致（毎回 0.80–0.87）は単発でも測り、報告する。

#### 比較結果（2026-09-30, Pegasus H100, 300 cells, seed 0/1/2）

**判定: D1 は両方で成立、D2 不成立 → 単発 feedback（最後の step の後に 1 回）を主結果にする。**
出力: `ISP-Platform/output/sf_stability_20260929/pegasus_out/aggregate/`（`mode_specificity.csv`,
`mode_comparison.json`, `seed_agreement_single_last.csv`）。単発の run は `single_seed{0,1,2}/`。

| 項目 | 毎回 feedback | 単発（最後の step 後） |
|---|---|---|
| OSKM の gain（細胞平均） | 0.303 / 0.321 / 0.333 | 0.271 / 0.252 / 0.273 |
| ランダム chain 5 本の gain 平均 | 0.283 / 0.304 / 0.309 | 0.217 / 0.208 / 0.217 |
| 特異的上乗せ Δspec [95% CI] | +0.019 [0.018, 0.020] / +0.018 [0.017, 0.019] / +0.024 [0.023, 0.025] | +0.054 [0.052, 0.056] / +0.043 [0.042, 0.045] / +0.056 [0.054, 0.057] |
| 特異的な割合 | 6.4% / 5.5% / 7.2% | 19.9% / 17.2% / 20.3% |
| Δspec 比（毎回 / 単発） | 0.36 / 0.41 / 0.43 | — |
| seed 間の最終 encoding の Spearman | 0.84 / 0.87 / 0.80 | 0.989 / 0.994 / 0.986 |
| Ordered rank-edit 終点との Spearman | 0.34–0.47 | 0.93–0.95 |

（seed 0 / 1 / 2 の順。seed 間の Spearman は seed 対 0-1 / 0-2 / 1-2。）

読み方:

- 毎回 feedback は終点を上げるが、上がる分はランダム chain でもほぼ同じだけ上がる。OSKM に特異的な上乗せは
  単発の 0.36–0.43 倍に減り、特異的な割合は約 20% から約 6% に下がる。feedback を重ねると、摂動に依らない
  goal 方向への引き寄せが特異的な信号を薄める
- 単発は decoder seed に対して最終 encoding がほぼ一致する（0.99）。毎回では 0.80–0.87
- 単発でも gain の約 8 割はランダム chain でも出る（decoder は endpoint 教師で学習しているため）。State-feedback の
  終点 shift は、ランダム chain の gain を差し引いた Δspec で報告する

#### 決定の更新（2026-09-30）

上の決定規則は単発を主結果に選んだが、単発（最後の step の後に 1 回）は最終 encoding だけで決まり、
Ordered rank-edit（= 逆順リストの multi-gene ISP）に下流の読み出しを 1 回足したものにすぎず、逐次効果を含まない。
逐次効果を扱うことが State-feedback の目的なので、**毎回 feedback を唯一の実装とし、ランダム chain の gain を
並行して取り差し引いた Δspec で報告する**。ランダム分が大きい（gain の約 94%）ことは限界として明記し、今後の
改善課題とする。単発・途中 1 回・回数上限の設定はコードから削除した（上の比較は commit `ee1ff49` で再現できる）。

上の結果のランダム chain は、teacher 遺伝子（体細胞・多能性細胞の両方で 5 細胞以上検出）から一様に引いたもので、
摂動の種類が OSKM と揃っていなかった（次節）。上の Δspec はこの点で割り引いて読む。

### ランダム chain の設計と検証（事前登録 2026-09-30、実行前に固定）

**旧設計の問題。** 開始細胞（somatic 300 細胞）で SOX2・POU5F1 は検出 0%（somatic 3,000 細胞でも 0）、
KLF4・MYC は 3.7%・7.3% で、検出されても順位は末尾側（正規化位置 0.92・0.86）。つまり OSKM の OE は
ほぼ全細胞で「encoding に無い遺伝子を先頭に挿入する」編集である。旧ランダム 20 遺伝子は検出 0–70%
（中央値 ~7%）、位置 0.10–0.95 で、多くは「すでにある遺伝子を前へ動かす」編集だった（例: DDR2 は 70% の細胞に
位置 0.22 で存在）。また母集団を teacher 遺伝子に限ったため、KLF4・SOX2・POU5F1 と同じ種類（片方の state に
ほぼ無い遺伝子）はそもそも引かれなかった。遺伝子名も記録していなかった。

**新設計**（`core/state_feedback/random_chains.py`）:

| 要素 | 内容 |
|---|---|
| 母集団 | モデル語彙の全遺伝子 token − 特殊 token − configured 遺伝子（teacher の検出フィルタを通らない遺伝子も含む） |
| 構造 | configured と同じ step 数・step あたり遺伝子数・型 |
| 層（configured 遺伝子ごと） | 開始細胞で検出 0 → 開始細胞のどれにも無い遺伝子。検出 d > 0 → 検出 d/2〜2d かつ位置（検出細胞の中央値）が ±0.1 以内。50 遺伝子未満なら ±0.2, ±0.3, 制限なしの順に広げ記録 |
| 抽出 | 層の中で一様、chain 間で非復元（層が足りなければ復元、記録）。configured 遺伝子を token 順に引くので、同じ乱数 seed なら step の順序によらず同じランダム遺伝子になる（24 order で同じランダム遺伝子の組を同じ順序で並べられる） |
| 独立な 2 抽出 | `random_seeds: [0, 1]` × 各 20 本 = 40 本 |
| 記録 | `random_chains.json`: 遺伝子名、各遺伝子の開始細胞検出・位置、goal 細胞の検出、teacher Δrank、層の大きさ、層に対するバランス（SMD） |
| CI | 細胞とランダム chain の二段 bootstrap（どのランダム遺伝子を引いたかのばらつきを含む） |

**検証**（configured = K→M→S→O、300 細胞、decoder seed 0/1/2。各基準は全 seed で成立すること）:

| ID | 問い | 基準 |
|---|---|---|
| R1 | 抽出は層の公平な標本か | configured 遺伝子ごとに、選ばれた 20 遺伝子の goal 細胞検出率と teacher Δrank の平均が層の平均から \|SMD\| ≤ 0.5（開始細胞の検出・位置は構造上揃う）。teacher Δrank は両 state で検出される遺伝子にしか無いので、開始細胞に無い層では goal 細胞検出率だけで判定する |
| R2 | 独立な 2 抽出が一致するか | ランダム gain の平均の差 \|s0 − s1\| ≤ 2·√(SE₀² + SE₁²)（SE は chain 間） |
| R3 | ランダム平均は十分に精密か | 40 本の平均の chain 間 SE ≤ 0.002（旧結果の Δspec の seed 間の幅 0.006 の 1/3） |

報告のみ: Δspec と二段 CI、leave-one-out null に対する位置（`null_empirical_p`）、本数に対するランダム平均の推移
（`random_running_mean.csv`）、旧ランダム chain の gain（0.283 / 0.304 / 0.309）との比較。

**決定規則:** R1–R3 が成立 → この設計で 24 order の検証に進む。本数は max(20, (chain 間 SD / 0.002)²)。
R1 不成立 → 層の定義を見直し（結果を見る前に理由を記録）、再抽出する。R2 または R3 不成立 → 本数を
(SD / 0.002)² に増やして再実行する。

#### 結果（2026-09-30, Pegasus H100, 300 cells, decoder seed 0/1/2, ランダム 2 抽出 × 20 本）

**判定: R1, R2, R3 すべて成立 → この設計で 24 order に進む。本数 20（(0.0058 / 0.002)² ≈ 9 < 20）。**
出力: `ISP-Platform/output/sf_stability_20260929/pegasus_out/matched_aggregate/`（`specificity.csv`,
`random_running_mean.csv`）、抽出は `matched_seed0/random_chains.json`（3 seed で同一の遺伝子）。

| 項目 | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| R1 \|SMD\| 最大（goal 検出 / teacher Δrank） | 0.36 / 0.30（両抽出、全遺伝子） | 同左 | 同左 |
| R2 ランダム gain 平均 s0 / s1（許容差） | 0.2911 / 0.2914（0.0033） | 0.3113 / 0.3118（0.0030） | 0.3164 / 0.3173（0.0035） |
| R3 40 本の平均の chain 間 SE | 0.0008 | 0.0008 | 0.0009 |
| OSKM の gain（細胞平均） | 0.3025 | 0.3214 | 0.3328 |
| Δspec [細胞 + chain の二段 95% CI] | +0.0113 [0.0094, 0.0130] | +0.0099 [0.0082, 0.0115] | +0.0159 [0.0140, 0.0178] |
| 特異的な割合 | 3.7% | 3.1% | 4.8% |
| OSKM の順位（41 chain 中、終点 gain の中央値） | 1 | 2 | 1 |

層: KLF4 259 遺伝子、MYC 333、SOX2・POU5F1 は開始細胞に無い 7,116 遺伝子（同じ層なので両者に同じ遺伝子が
別 chain で当たることがあり、80 本中の重複は s0 で 2、s1 で 0）。

読み方:

- 抽出は層に対して偏りがなく、独立な 2 抽出のランダム平均は 0.001 以内で一致する。ランダム平均は 5 本で
  すでに ±0.003 以内に落ち着く（`random_running_mean.csv`）
- 摂動の種類を揃えると、ランダム chain の gain は旧設計より上がり（0.283–0.309 → 0.291–0.317）、Δspec は
  旧設計の約 6 割（+0.018–0.024 → +0.010–0.016）になる。旧設計は「無い遺伝子の挿入」という OSKM の編集の
  大きさをランダム側で再現しておらず、特異性を過大に見積もっていた
- Δspec の CI はすべて 0 を含まないが、特異的な割合は 3–5% で、gain のほとんどは遺伝子に依らない
- S3（OSKM の gain がどのランダム chain よりも大きい）は seed 1 で不成立: random_s1_16（HIF3A, HES1,
  NOTCH2NLA, OR4C3）が OSKM を上回り、他の seed でも 2 位。OSKM は 40 本の分布の上端（leave-one-out null の
  経験 p 0.024–0.049）にある


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
  + λ₄ · L_smooth          (regularize extreme Δr)
```

当初あった λ₃ · L_identity（ctrl→ctrl で Δr ≈ 0）は、decoder の入力を Δh のみ（bias なし）にしたことで構造的に常に 0 になるため削除した。

`L_pair` は **Δr（変位）の順序**に対して定義する。絶対 priority (`base_rank_norm + Δr`) の順序に対して定義してはならない — `base_rank_norm` を tanh で増幅すれば満たせてしまい、Δh 由来の情報が失われる（上記 decoder 節参照）。

実装既定値 (`state_feedback.decoder`): `lam_huber: 1.0`, `lam_pair: 0.5`, `lam_smooth: 0.01`, Huber δ = 0.05, Adam。


## scPRINT-2 reference (design parallel, not drop-in)

Found under `~/20260323scPRINT-2/` (also `~/20260531scPRINT2andPanOrganseq/scPRINT-2-private/`):

- Spec: `docs/ISP_v2_spec.md`
- `predict_expression` → `pred_start.h5ad` / `pred_pert.h5ad` / `delta_ranked_genes.csv`
- Enrichment: `services/run_isp_enrichment.py` (gseapy prerank)

scPRINT predicts expression via ZINB mean (`scprint_mu`), then ranks deltas. This is analogous to MLP Δrank + expression decoder — natural for scPRINT's architecture but inapplicable to Geneformer (no expression decoder). The residual Δrank approach above achieves a comparable pipeline shape without requiring an expression decoder.
