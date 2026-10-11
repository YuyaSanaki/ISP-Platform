### English

Primary evaluation on held-out genes: compares identity, random, norm, delta_mlm, base_rank and linear_deltarank by rank correlation with the teacher. PASS needs the decoder to beat `norm` and `delta_mlm` with bootstrap CIs excluding 0 **and** a partial ρ given base rank with CI > 0. Writes `direction_fidelity/`. Optional perturbation specificity compares Δh from other gene sets and random draws.

### 日本語

学習に使わなかった遺伝子での主評価です。identity、random、norm、delta_mlm、base_rank、linear_deltarank を、教師との順位相関で比べます。PASS には、デコーダーが `norm` と `delta_mlm` を上回り、その差のブートストラップ信頼区間が 0 を含まないこと、**かつ**元の順位で調整した偏相関 ρ の信頼区間が 0 より上であることが必要です。結果は `direction_fidelity/` に書きます。任意の摂動特異性評価では、ほかの遺伝子セットやランダムに選んだ遺伝子の Δh と比べます。
