### English

Optional downstream plots: cluster/cell-type annotation, joint overlays, cell-type trajectories, L2 by coarse cell type. Switched on by `postprocess.enabled` (`core/postprocess_switch.py`), the same switch that runs rank enrichment; cells left Unknown or Ambiguous are drawn in grey in joint overlays. Joint overlays use the run's `umap.pca_components`, so they match the main plot's projection; `annotate_isp_umap_celltypes --refresh-overlays` takes `--gene` or detects it from the run folder.

### 日本語

任意の後処理の図です。クラスタと細胞種の付与、重ね合わせ図、細胞種ごとの軌跡、粗い細胞種ごとの L2。`postprocess.enabled`（`core/postprocess_switch.py`）でオンになり、rank enrichment と同じスイッチです。重ね合わせ図では Unknown や Ambiguous のままの細胞を灰色で描きます。重ね合わせ図は実行の `umap.pca_components` を使うので、メインの図と同じ投影になります。`annotate_isp_umap_celltypes --refresh-overlays` は `--gene` を受け取るか、実行フォルダから遺伝子を検出します。
