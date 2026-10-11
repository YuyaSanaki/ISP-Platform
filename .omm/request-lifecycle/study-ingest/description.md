### English

**Zip upload**: study name normalised, zip read into RAM, extracted (unsafe paths rejected) to
`{WEBUI_WORKSPACE}/uploads/<session>/<study>/`, macOS junk removed, wrapper folder lifted. **URL import**: http(s) only;
Google Drive / Dropbox links rewritten; GEO GSE → NCBI FTP supplementary; CELLxGENE via curation API; `.h5ad` rejected;
8 GiB / 600 s cap; tar repacked to zip.

### 日本語

**zip アップロード**: study 名を正規化し、zip をメモリに読み込んで `{WEBUI_WORKSPACE}/uploads/<session>/<study>/` に展開します（危険なパスは拒否）。
macOS の不要ファイルを消し、外側の包みフォルダを取り除きます。**URL 取り込み**: http(s) のみ。Google Drive / Dropbox のリンクを書き換え、
GEO の GSE は NCBI FTP の補足ファイルへ、CELLxGENE は curation API 経由で解決します。`.h5ad` は拒否、上限 8 GiB / 600 秒、tar は zip に詰め直します。
