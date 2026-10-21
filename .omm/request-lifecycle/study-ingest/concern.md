### English

Residual limits after the fixes: `st.file_uploader` itself keeps an uploaded zip in server memory (Streamlit behaviour; the app now copies it to disk in blocks without a second copy), so very large studies are better imported with **Download from URL**. The public-address check resolves the host before urllib connects, so a DNS answer that changes between the two lookups (rebinding) is not excluded.

### 日本語

修正後に残る制限: `st.file_uploader` 自体がアップロードされた zip をサーバーのメモリに置きます（Streamlit の仕様。アプリはそれを2つ目のコピーを作らずにブロック単位でディスクに写すようになりました）。とても大きな study は **Download from URL** で取り込むほうが安全です。公開アドレスの確認は urllib が接続する前にホスト名を引くので、2回の名前解決の間に DNS の答えが変わる場合（DNS rebinding）は防げません。
