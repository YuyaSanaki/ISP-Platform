# Repository operations — two-repo model

Last updated: 2026-09-02 (split started)

This document defines how **Geneformer Platform** and **Geneformer Platform Analysis** are split, versioned, and maintained. **The same file lives in both repositories** so either checkout remains the source of truth for the relationship.

---

## 1. Overview

| Repository | Visibility | Role |
| ---------- | ---------- | ---- |
| **[Geneformer Platform](https://github.com/YuyaSanaki/Geneformer-Platform)** | **Public** | Reusable software: `core/`, Web UI, Docker, tests, user-facing docs. **Current release: v1.0.0.** |
| **[Geneformer Platform Analysis](https://github.com/YuyaSanaki/Geneformer-Platform-Analysis)** | **Private** | Papers and lab projects: BBRC, Asano, `analysis/`, `output/`, `data/`, manuscripts, `ToDo.txt`, `scripts/bbrc_oskm/`. |

**Principle:** Analysis **depends on** Platform via a **pinned Docker image tag** (`geneformer-platform:v1.0.0`). Platform does **not** depend on Analysis.

```
Geneformer Platform v1.0.0     ← public; Docker image geneformer-platform:v1.0.0
        ↑
        │  GENEFORMER_PLATFORM_IMAGE=geneformer-platform:v1.0.0
        │
Geneformer Platform Analysis   ← private; may version data, outputs, drafts in git
```

---

## 2. What belongs in each repo

### Geneformer Platform (public)

| Include | Exclude |
| ------- | ------- |
| `core/`, `webui/`, `Dockerfile`, `docker-compose.yml` | `analysis/`, paper `output/` snapshots |
| `tests/`, platform `scripts/` (download assets, smoke matrix) | `docs/bbrc/`, `docs/projects/` |
| User docs: `docs/tokenization.md`, `docs/web-ui.md`, … | `ToDo.txt`, `GEMINI.md`, `docs/HANDOFF.md` |
| Generic example configs (`core/config/` → smoke paths) | `scripts/bbrc_oskm/` |
| `VERSION`, `CHANGELOG.md`, `models/README.md` | Manuscript drafts |

### Geneformer Platform Analysis (private)

| Include in git | Still exclude |
| -------------- | ------------- |
| `docs/bbrc/`, `docs/projects/`, `scripts/bbrc_oskm/` | `.env`, API keys, `scripts/bbrc_oskm/bbrc_runpod.env` |
| `analysis/`, `output/`, `data/` (private; not leaked externally) | `__pycache__/`, `.venv*/`, large model weights under `models/` |
| `ToDo.txt`, manuscripts, audits | Secrets and credentials |

**Transition note:** Analysis may still contain a copy of `core/` / `webui/` while compose uses `.:/app` bind-mounts. Runtime baseline is the **Docker image tag** in `platform_pin.yaml` / `.env`. Remove duplicate Platform source from Analysis once mounts are image-only.

---

## 3. How Analysis uses Platform (canonical: Docker image tag)

1. Platform release **v1.0.0** → build and tag locally or in CI:

   ```bash
   cd ~/Geneformer-Platform
   docker compose --profile build build geneformer-platform
   docker tag geneformer-platform:latest geneformer-platform:v1.0.0
   ```

2. Analysis sets in `.env` (see `.env.example`):

   ```bash
   GENEFORMER_PLATFORM_IMAGE=geneformer-platform:v1.0.0
   ```

3. `docker-compose.yml` uses `${GENEFORMER_PLATFORM_IMAGE:-geneformer-platform:v1.0.0}` for all GPU/CPU services.

4. Record the pin in `platform_pin.yaml` at the Analysis repo root.

**Do not** merge Platform bugfixes by hand into Analysis copies of `core/`. Fix Platform, release a new tag, bump the image pin in Analysis.

---

## 4. Versioning and reproducibility

### Platform

- **v1.0.0** = first public release (2026-09-02 split).
- SemVer for subsequent releases; update `VERSION` + `CHANGELOG.md`.
- Git tag: `v1.0.0` matches Docker tag `geneformer-platform:v1.0.0`.

### Analysis

- `platform_pin.yaml` at repo root (canonical pin).
- Per-project notes under `analysis/<project>/` as needed.
- Methods / Supplement: cite **Platform v1.0.0** + Analysis commit (private repo URL optional).

---

## 5. Git and history

| Repo | History policy |
| ---- | -------------- |
| **Platform** | **Clean initial commit** at split (no `ToDo.txt`, no BBRC drafts in history). |
| **Analysis** | Existing monolith history stays **private**. May commit `analysis/`, `output/`, `data/`, `ToDo.txt`, manuscripts. |

Platform pre-push check:

```bash
git ls-files | rg -i 'bbrc|asano|todo|manuscript|handoff'   # should be empty
git log --all -- ToDo.txt                                     # should be empty
```

---

## 6. Local directory layout (target)

| Path | Repo |
| ---- | ---- |
| `~/Geneformer-Platform` | Public Platform (v1.0.0) |
| `~/Geneformer-Platform-Analysis` or `~/20260624Geneformer-Platform` | Private Analysis (rename when convenient) |

---

## 7. Migration checklist

- [x] Policy documented (`docs/repository_operations.md` in both repos).
- [x] Platform tree exported to `~/Geneformer-Platform` (no BBRC / analysis).
- [x] Platform default configs genericized (smoke_mouse paths).
- [x] Platform `VERSION` / `CHANGELOG` v1.0.0.
- [x] Analysis `platform_pin.yaml` + `.env.example`.
- [ ] `git init` + tag `v1.0.0` in Platform; push to new public GitHub repo.
- [ ] Build `geneformer-platform:v1.0.0` image.
- [ ] Rename GitHub `Geneformer-Platform` → `Geneformer-Platform-Analysis` (private).
- [ ] Push Analysis with updated `.gitignore` (track data/output/analysis).
- [ ] Smoke test Analysis job with pinned image only.

---

## 8. When this policy changes

1. Edit this file in **both** repositories together.
2. Update the “Last updated” line.
3. Bump `platform_pin.yaml` and rebuild Docker when Platform releases.
