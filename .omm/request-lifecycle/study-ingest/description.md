### English

**Zip upload**: study name normalised, the uploaded zip copied to a temporary file in blocks and extracted from there
(unsafe paths rejected) to `{WEBUI_WORKSPACE}/uploads/<session>/<study>/`, macOS junk removed, wrapper folder lifted;
re-imports are skipped by name, size and SHA-256. **URL import**: http(s) only, public addresses only (the host and every
redirect target must resolve to global IPs unless `ISP_URL_IMPORT_ALLOW_PRIVATE=1`); Google Drive / Dropbox links
rewritten; GEO GSE → NCBI FTP supplementary; CELLxGENE via curation API; `.h5ad` rejected; streamed to disk with an
8 GiB / 600 s cap; tar repacked to zip.

### 日本語

**zip アップロード**: study 名を正規化し、アップロードされた zip をブロック単位で一時ファイルに写してそこから `{WEBUI_WORKSPACE}/uploads/<session>/<study>/` に展開します（危険なパスは拒否）。
macOS の不要ファイルを消し、外側の包みフォルダを取り除きます。取り込み済みかどうかは名前、サイズ、SHA-256 で判定します。**URL 取り込み**: http(s) のみ、公開アドレスのみ
（`ISP_URL_IMPORT_ALLOW_PRIVATE=1` でない限り、ホストとすべてのリダイレクト先がグローバルな IP に解決される必要がある）。Google Drive / Dropbox のリンクを書き換え、
GEO の GSE は NCBI FTP の補足ファイルへ、CELLxGENE は curation API 経由で解決します。`.h5ad` は拒否、ディスクに逐次書き込み、上限 8 GiB / 600 秒、tar は zip に詰め直します。
