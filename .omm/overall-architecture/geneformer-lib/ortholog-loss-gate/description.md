### English

Fail-closed check for critical gene sets (e.g. OSKM) before tokenizing. **Block** when a critical gene is present in the
raw input but dropped by the mapping policy (or provenance hash missing / mapped % below an opt-in floor); **Warn** when
it is absent from the input; otherwise **Pass**. On Block it writes `ortholog_approval_request.yaml` (`status: pending`,
`request_id` = sha256 of the blocked run) and the tokenizer exits with code 2. An `approval_record.yaml` with
`status: approved` must match the recomputed hashes. Decisions: stop (A), curated_bridge (B, re-evaluate with overlay),
sensitivity_best_of_n (C, not auto-run), mark_non_critical (D).

### 日本語

トークン化の前に重要遺伝子セット（OSKM など）を確認する、安全側に倒れるチェックです。重要遺伝子が入力にあるのに変換方針で落ちる場合
（または由来ハッシュの欠落、オプトインした下限を下回る変換率）は **Block**、入力にそもそもない場合は **Warn**、それ以外は **Pass**。
Block では `ortholog_approval_request.yaml`（`status: pending`、`request_id` は止まった実行の sha256）を書き、トークナイザーは終了コード 2 で止まります。
承認には `status: approved` の `approval_record.yaml` が必要で、ハッシュが再計算値と一致しなければなりません。
判断は stop（A）、curated_bridge（B、オーバーレイ付きで再評価）、sensitivity_best_of_n（C、自動実行なし）、mark_non_critical（D）。
