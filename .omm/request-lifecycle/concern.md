### English

- Job state lives only in the browser session (`st.session_state.active_proc`): no global lock, so two sessions can run jobs at once; tracking is lost on reload.
- No cancel / kill anywhere in `webui/`.
- No authentication; "Approved by" is free text.

### 日本語

- ジョブの状態はブラウザのセッション（`st.session_state.active_proc`）にしかありません。全体のロックがないため、2つのセッションから同時にジョブを動かせ、再読み込みすると追跡が失われます。
- `webui/` のどこにもキャンセル・停止の処理がありません。
- 認証がなく、「Approved by」は自由入力です。
