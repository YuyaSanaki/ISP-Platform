# State-feedback ISP — 逐次 KO の反実仮想と検証設計（2026-10-01）

状態: **実装済み**（commit `bdd8cac`）。判定基準 K1–K4 は
[state_feedback_decode_methods.md](state_feedback_decode_methods.md) の「逐次 KO の検証」に事前登録した。OE 版の潜在反応 + Δspec（`core/state_feedback/placebo_contrast.py`,
commit `8749a29`）に KO の分岐を足す。親文書:
[state_feedback_decode_methods.md](state_feedback_decode_methods.md) の「方針（2026-10-01 決定）」。

目的: 論文（BBRC iPS / 体細胞データのみ, `GSE147564_ft6k_qc_human_native_0`,
`v2_ft_human_native/all_run1`）で、State-feedback ISP が **KO を含む逐次 chain** に対応することを示す。
1 step の KO は feedback 後に次の摂動が無く、Ordered rank-edit（普通の ISP）に読み出しを 1 回足した
ものにすぎない（2026-09-30 の決定と同じ理由）ので、すべて 2 step 以上にする。Asano データはこの論文
では使わない。

## 1. 検証する 3 本の chain

| ID | chain | 方向（start → goal, 教師） | 期待 | 開始細胞 |
| --- | --- | --- | --- | --- |
| KO-A | POU5F1 KO → L1TD1 KO | pluripotent → somatic（教師 = 観測 somatic） | 動く（プラセボより大きく somatic 側へ） | pluripotent, 長さ降順の先頭 N |
| KO-B | TP53 KO → KLF4 → MYC → SOX2 → POU5F1（OE） | somatic → pluripotent（教師 = 観測 pluripotent, OE の X3 と同じ） | OSKM に上乗せする | **TP53 が検出された somatic 細胞すべて（116 / 3,000）** |
| KO-C | DNMT3B KO → DPPA4 KO | pluripotent → somatic（教師 = 観測 somatic） | 動かない（プラセボの分布の中） | KO-A と同じ細胞 |

生物学的な根拠（引用前にすべて確認する）:

- KO-A: POU5F1 の KD でヒト ES 細胞が分化（Matin 2004; Hay 2004; Zaehres 2005）。L1TD1 はヒト ES 細胞の
  自己複製に必須で LIN28 と結合（Närvä 2012）。
- KO-B: p53 の抑制で iPS 初期化の効率が上がる（Hong / Kawamura / Marion / Utikal / Li 2009; Zhao 2008）。
- KO-C: DNMT3A/3B を KO したヒト ES 細胞は自己複製し多能性を保つ（Liao 2015）。Dppa4 はマウス ES 細胞の
  多能性に不要（Madan 2009, マウスのみ）。DNMT3B は pluripotent 細胞でほぼ先頭（正規化位置 0.003）に
  いるので、「エンコーディングを大きく変えても生物学的に効かない KO では動かない」ことの対照になる。
- 注意: KO された ES 細胞は栄養外胚葉・内胚葉へ分化し、線維芽細胞にはならない。2 クラスの分類器では
  「多能性から離れる」ことが somatic 側への shift として読まれるので、KO-A の shift は控えめでありうる。

## 2. BBRC で確認した事実（2026-10-01）

| chain | 開始細胞 | 両 KO 遺伝子がある細胞（処置細胞） | 評価用プラセボ 30 本のうち、処置細胞にプラセボ遺伝子がそろう本数 |
| --- | --- | --- | --- |
| KO-A | pluripotent 先頭 300（長さ中央値 4096） | 257（86%） | 中央値 17, 最小 10 |
| KO-A | pluripotent 先頭 100 | 85 | 中央値 17, 最小 7 |
| KO-C | pluripotent 先頭 300 | 294（98%） | 中央値 26, 最小 14 |
| KO-C | pluripotent 先頭 100 | 99 | 中央値 27, 最小 22 |
| KO-B | somatic 先頭 300 | 33（11%） | 中央値 4, 最小 0（25 細胞で 5 本未満） → 不可 |
| KO-B | **TP53 検出 somatic 116 細胞**（層もこの細胞で作る） | 116 | 中央値 22, 最小 7 |

- 層（`random_chains.stratum`, 検出 d/2–2d かつ位置 ±0.1）: POU5F1 1,082 遺伝子（pluripotent 300）。他の
  遺伝子も 50 以上。
- TP53 は somatic 全 3,000 細胞の 3.9% にしか検出されない。開始細胞を TP53 検出細胞に限るのは、KO が
  意味を持つ細胞だけで比べるため（KO-B の比較はすべてこの 116 細胞の中で行う）。
- 却下した候補（記録）: 1 step の KO（逐次でない）。somatic → pluripotent の他の障壁遺伝子（CDKN2A 12%,
  DOT1L 18%, MBD3 24%, CDKN1A 39% だが位置 0.90）。体細胞マーカー（COL1A1, THY1）の KO（主張が弱い）。
  OSNL の KO（4 遺伝子がそろう細胞 4%）。

## 3. 反実仮想（OE と KO で一つの原則）

各 rerank の反実仮想エンコーディングは、**現在のエンコーディングで chain の因子の編集をすべて取り消し、
同じ種類の編集をマッチしたプラセボ遺伝子に施したもの**とする。m_g = 推定用プラセボについての
反実仮想 Δh の平均、decoder への入力は Δh − m_g。

| 編集 | 取り消し | プラセボに施す |
| --- | --- | --- |
| OE（実装済み, `swap_tokens`） | 因子の slot から因子を外す | その slot にプラセボ遺伝子を置く（別の位置にあれば除く） |
| KO（本案, `undo_delete_redo`） | 因子を **開始エンコーディングで直前にあった遺伝子の直後** に戻す | プラセボ遺伝子を削除する |

### 3.1 KO の反実仮想 `undo_delete_redo`

1 つの推定用プラセボ p（slot ごとの遺伝子 q_s）について、現在のエンコーディング X_t から:

1. 削除された因子 T_s を、開始エンコーディング X_0 での位置の前から順に戻す。T_s の anchor は、X_0 で
   T_s より前にあり、作業中のエンコーディングに存在し、p のどの遺伝子でもない遺伝子のうち T_s に最も
   近いもの。anchor の直後に挿入する（無ければ OE で先頭に固定した遺伝子の直後、それも無ければ先頭）。
2. OE の slot は既存の `swap_tokens` と同じ扱い。
3. KO の slot のプラセボ遺伝子 q_s を削除する。

性質:

- **feedback 前は厳密に一致する。** X_t = X_0 \ {T_1, T_2}（rerank 前）なら、結果は X_0 \ {q_1, q_2}
  （プラセボを最初から KO したエンコーディング）と完全に一致する（q が T の直前にあっても anchor の探索で
  飛ばすので成り立つ）。decoder の学習側の対比（at-once のエンコーディング）と同じ反実仮想になる。
- **逐次 chain では、rerank で動いた anchor の近傍に T を戻す。** KO-A / KO-C の 2 回目の feedback、
  KO-B の 2〜5 回目の feedback では、T が無かった間の rerank で anchor が動いた分に T が追随する。
- 長さは X_t と同じ。遺伝子集合は X_t − {q_s} + {T_s}。

却下した案: slot 交換（q の位置に T を置く; feedback 前でも at-once の反実仮想と一致しない）、開始時の
正規化位置への挿入（rerank 後の近傍を無視する）、decoder の priority で位置を決める（反実仮想が decoder
に依存して循環する）。

### 3.2 編集が起きなかった細胞（null の保証）

KO の因子がその細胞に無かった（T_s ∉ X_0,i）slot では、**プラセボの編集も施さない**。全 slot がそうなら
その細胞は m_g = 0、Δh = 0、変位 0。OE は常にエンコーディングを変えるので、この規則は OE に影響しない。

### 3.3 プラセボ

- 層: `random_chains.stratum`（OE と同じ関数）。母集団は語彙全体 − 特殊 token − chain の遺伝子。
- 評価用（Δspec）: `draw_random_chains(..., seed=0)`、30 本。slot ごとにマッチ。
  - KO-A / KO-C: 両 slot をプラセボに置き換えた chain。
  - KO-B: **KO の slot だけ**をプラセボに置き換え、OSKM はそのまま（q_p KO → K → M → S → O）。
    推定量は「OSKM の下での TP53 KO の上乗せ」で、step 数（feedback 5 回）も揃う。
- 推定用（m_g と decoder 学習）: 既存の `draw_estimation_placebos`（`estimation_draw_seed: 1`）。
  KO の run では `n_estimation: 20`（KO-A で 10 本だと処置細胞あたり 6 本程度しか存在しないため。
  `ESTIMATION_POOL` = 20 なので 10 本は 20 本の先頭と入れ子）。chain と遺伝子を共有するプラセボは、
  OE 実装と同じくその chain の対比から外す。
- 推定用プラセボの KO 遺伝子がその細胞の現在のエンコーディングに無ければ、その細胞の m_g の平均から外す。
  使えるプラセボが 3 本未満の細胞は解析から外し、数を報告する。

### 3.4 decoder の学習

- スケジュールごとに decoder を学習する（KO-A, KO-B, KO-C で別々。各 decoder をその chain のプラセボにも
  使う。OE と同じ方針）。seed 0/1/2。
- 入力: at-once のエンコーディングでの Δh − mean_p Δh_p（同じ細胞・同じ遺伝子）。
- **学習サンプルは処置細胞に限る**（KO-A / KO-C は両 KO 遺伝子がある細胞、KO-B は開始細胞すべて）。
  因子が無い細胞では Δh = 0 で、対比が −(プラセボの平均 Δh) になり、プラセボ KO の効果の符号反転を
  教師に当てはめてしまう。
- 比較対照: 評価用プラセボ chain で学習した decoder（10 本）の held-out 遺伝子 Spearman（selectivity）。

### 3.5 結果側（Δspec）

- 細胞ごと: τ_i = gain_i(因子 chain) − mean_{p ∈ P_i} gain_i(p)。gain は同じ chain の Ordered rank-edit
  終点からの上乗せ。P_i = その細胞にプラセボの KO 遺伝子がすべてあった評価用 chain（無い chain の gain は
  構造的に小さく、含めると因子が有利になる）。
- 細胞集合: KO-A / KO-C は処置細胞、KO-B は 116 細胞すべて。プラセボ自身の評価（leave-one-out, z, 順位）も
  同じ細胞集合・同じ条件付き平均。
- KO-A と KO-C の直接比較: 両方の処置細胞（4 遺伝子すべてがある細胞）で τ_A,i − τ_C,i（細胞で対、
  bootstrap CI）。
- KO-B の補助: 同じ 116 細胞で gain(TP53 → OSKM) − gain(OSKM のみ)（feedback 回数が 5 対 4 で違うので
  参考値。主の比較は KO の slot だけのプラセボ）。

## 4. 判定基準（投入前に固定する案。3 seed すべてで成立すること）

| ID | chain | 基準 |
| --- | --- | --- |
| K1 | KO-A | τ > 0、評価用プラセボ 30 本に対して z ≥ 3、31 本中 1 位 |
| K2 | KO-B | τ > 0（KO の slot だけのプラセボに対して）、z ≥ 2 |
| K3 | KO-C | \|z\| < 2（プラセボの分布の中）、かつ τ_A − τ_C の CI が 0 を含まない（KO-A > KO-C） |
| K4 | 全 chain | 摂動なし chain・因子が無い細胞で変位 0 / Spearman 1.0 |

- KO-B は処置細胞が 116 で、p53 の効果は「効率の上昇」なので期待効果が小さい。基準を z ≥ 2 に緩めて
  いる理由として記録する。不成立なら数値をそのまま報告する。
- K3 は「効果が無いこと」の証明ではない。「生物学的に効かないはずの KO が、マッチしたプラセボと区別
  できない」ことと、「効くはずの KO より小さい」ことの 2 点として書く。

## 5. 計算量（見積もり）と細胞数

実測: somatic 300 細胞、OSKM 4 step、V2 ループ 1 本 815–1,059 秒（1 step 約 3.5 分）、Ordered rank-edit
1 step 約 65 秒（`x7c_methods_n300`, 2026-10-01）。pluripotent の先頭細胞は 4096 トークンで、somatic
（中央値約 2,100）の約 2 倍の長さなので forward は 2–3 倍と仮定した。

| chain | n | 内訳 | ノード時間 |
| --- | --- | --- | --- |
| KO-A | 300 | 31 chain × 3 seed × V2 約 17 分 + rank-edit 31 × 約 5 分 + 特徴量・decoder | 約 32 |
| KO-C | 300 | 同上 | 約 32 |
| KO-A | 100 | 約 1/3 | 約 11 |
| KO-C | 100 | 約 1/3 | 約 11 |
| KO-B | 116 | 32 chain（TP53 + 30 プラセボ + OSKM のみ）× 3 seed × V2 約 7 分 + rank-edit + decoder | 約 14 |

- 全部 n=300 なら約 78 ノード時間、KO-A / KO-C を n=100 にすると約 36 ノード時間。
- 残りの計算予算では、KO-A / KO-C は **n=100** でないと収まらない見込み。分散の分解（n=50 スクリーニング）では推定値の
  ばらつきの約 99% がプラセボ chain 間の差で、細胞数を減らしても z と CI はほとんど変わらない。
- 最初の job（seed 0 の数 chain）で実測し、予算表を更新してから残りを投入する。

## 6. 実装の形（OE の経路を変えない）

- `core/state_feedback/placebo_contrast.py`:
  - `check_overexpress_only` を「OE と KO 以外の型はエラー」に置き換える（KO を含まない config の挙動は
    変わらない）。
  - 新関数 `undo_delete_redo(ids_t, ids_0, deleted_slots, placebo, pinned)`。`swap_tokens` には触れない。
  - `placebo_mediator` / `collect_contrast_training_samples` で、chain に KO の step があるときだけ
    KO の分岐（§3.1, §3.2, 処置細胞の制限）を通す。
- 評価用プラセボの「KO の slot だけ置き換える」指定（KO-B）: `state_feedback.stability` に、置き換える
  step の番号のリストを置く（既定は全 step = 現行の挙動）。
- 開始細胞を「指定遺伝子が検出された細胞」に限る指定（KO-B）: `isp` か `state_feedback` に
  `require_detected: [TP53]` を置く（既定は無し = 現行の長さ降順の選択）。
- Δspec の細胞集合と条件付きプラセボ平均（§3.5）は、chain に KO があるときだけ有効にする。OE だけの
  chain では全細胞・全プラセボになり、結果は変わらない。
- config 例: `core/config/state_feedback_ko_pou5f1_l1td1.yaml`, `..._tp53_oskm.yaml`,
  `..._dnmt3b_dppa4.yaml`。

## 7. テスト

| テスト | 期待 |
| --- | --- |
| feedback 前の一致 | `undo_delete_redo(X_0 \ {T_1,T_2}, X_0, ...) == X_0 \ {q_1,q_2}`（q が T の前・後・直前、T が先頭の各場合） |
| 逐次での追随 | rerank 後のエンコーディングで、T が anchor の直後に戻る |
| 長さと遺伝子集合 | 長さ = X_t、遺伝子集合 = X_t − {q} + {T} |
| 因子が無い細胞 | m_g = 0、変位 0 |
| 摂動なし chain | 変位 0 / Spearman 1.0 |
| OE と KO の混在（KO-B） | KO の slot は anchor で戻り、OE の slot は `swap_tokens` と同じ結果 |
| OE の基準データ | KO の分岐を足す前に保存した OE run（数細胞、1 chain）の細胞ごとの出力と完全一致 |

## 8. 決定事項と未決定

決定（2026-10-01）:

1. 3 本の chain（KO-A, KO-B, KO-C）で、State-feedback の逐次 KO 対応を示す。
2. KO-B の開始細胞は TP53 が検出された somatic 細胞すべて（116）。
3. KO の因子は anchor 挿入で戻す。slot 交換は感度分析にもしない。
4. decoder の学習と Δspec は処置細胞に限る。
5. 陽性対照・陰性対照は KO-A / KO-C 自身（期待が逆向きの 2 本）。Asano は使わない。

未決定:

- KO-A / KO-C の細胞数（n=300 は予算に収まらない見込み。推奨 n=100）。
- 判定基準 K1–K4 の確定と事前登録（`docs/state_feedback_decode_methods.md` に、投入前に commit）。
