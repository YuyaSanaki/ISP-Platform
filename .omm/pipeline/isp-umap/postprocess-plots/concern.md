### English

Joint overlays recompute UMAP without PCA, so their coordinates differ from the main PCA(50) plot. `annotate_isp_umap_celltypes --refresh-overlays` hard-codes gene `Igfbp2`; other genes only log a warning.

### 日本語

重ね合わせ図は PCA なしで UMAP を計算し直すため、メインの PCA(50) の図と座標が一致しません。`annotate_isp_umap_celltypes --refresh-overlays` は遺伝子 `Igfbp2` を決め打ちしており、ほかの遺伝子では警告が出るだけです。
