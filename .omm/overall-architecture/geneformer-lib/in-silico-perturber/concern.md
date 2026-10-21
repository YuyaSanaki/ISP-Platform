### English

By default, centroids still use the first `max_ncells` cells pooled over all states, so published scores are unchanged. The run now prints the cells per state and warns when a state has under 10% of the largest state's cells. `isp.max_ncells_per_state` (opt-in) samples each state separately, as the State-feedback runners do; it changes centroids and therefore scores.

### 日本語

既定では、セントロイドは今もすべての状態をまとめた先頭 `max_ncells` 個の細胞を使うので、公表済みのスコアは変わりません。実行時に状態ごとの細胞数を表示し、ある状態の細胞が最大の状態の 10% 未満なら警告します。`isp.max_ncells_per_state`（任意で有効化）を設定すると、State-feedback の実行と同じく状態ごとに別々に抽出します。セントロイドが変わるのでスコアも変わります。
