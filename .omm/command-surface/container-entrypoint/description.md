### English

On aarch64 sets `LD_PRELOAD` for libgomp; copies baked models into `/app/models/<name>` and dict files into `/app/core/geneformer/dicts` only when missing (host files win). `SKIP_MODEL_SEED=1` disables seeding. Ends with `exec "$@"`.

### 日本語

aarch64 では libgomp 用に `LD_PRELOAD` を設定します。焼き込んだモデルを `/app/models/<name>` に、辞書を `/app/core/geneformer/dicts` に、無い場合だけコピーします（ホスト側のファイルが優先）。`SKIP_MODEL_SEED=1` でコピーを止められます。最後に `exec "$@"`。
