### English

ISP Platform is a monorepo that runs Geneformer workflows (tokenize, fine-tune, in silico perturbation (ISP), ISP UMAP and
State-feedback multi-step ISP) for mouse and human scRNA-seq, with ortholog-based cross-species conversion.
Two front doors drive the same code: the Streamlit **Web UI** and the **CLI / Docker Compose** services.
Core logic lives in `core/`; the Web UI only writes YAML and spawns `core/run_*.py` subprocesses.

### 日本語

ISP Platform は、マウス・ヒトの scRNA-seq に対して Geneformer のワークフロー（トークン化、ファインチューニング、
in silico 摂動（ISP）、ISP UMAP、State-feedback 多段階 ISP）を実行するモノレポです。オルソログ変換で種をまたいだ解析もできます。
入口は Streamlit の **Web UI** と **CLI / Docker Compose** の2つで、どちらも同じコードを動かします。
処理本体は `core/` にあり、Web UI は YAML を書いて `core/run_*.py` をサブプロセスで起動するだけです。
