FROM nvcr.io/nvidia/pytorch:25.03-py3

ENV DEBIAN_FRONTEND=noninteractive
# core = geneformer + runners; contracts = shared layout; webui = Streamlit only
ENV PYTHONPATH=/app/core:/app/contracts:/app/webui

WORKDIR /app

# Install uv directly from its official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies via uv.
# We strip out torch, torchvision, torchaudio, tbb, triton and nvidia-* because the NGC PyTorch image already provides highly-optimized versions.
# Remove the EXTERNALLY-MANAGED marker so uv/pip can install system-wide in the container.
# Strip strict version pins (==x.y.z) since requirements.txt was created for Python 3.8
# and many pins are incompatible with Python 3.12. We keep >=/>/>= constraints.
RUN rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED && \
    grep -v "^nvidia-" requirements.txt | grep -v "^torch" | grep -v "^tbb" | grep -v "^triton" | grep -v "^tensorflow" | grep -v "^keras" | \
    sed 's/==.*//' > req_filtered.txt && \
    uv pip install --system -r req_filtered.txt && \
    # anndata>=0.12 / scverse-misc need typing_extensions.Format (added in 4.13).
    # NGC base may already ship 4.12.x; force upgrade so imports do not break.
    uv pip install --system "transformers>=4.40,<5" gdown "huggingface_hub>=0.23" "typing-extensions>=4.13"

# Copy the rest of the project (models/ and *.pkl excluded via .dockerignore)
COPY . .

RUN chmod +x /app/scripts/container-entrypoint.sh \
             /app/scripts/download_build_assets.sh \
             /app/scripts/download_mouse_geneformer.sh \
             /app/scripts/download_human_geneformer_v2_104m.sh \
             /app/scripts/download_mouse_human_orthologs.sh \
             /app/scripts/download_drosophila_orthologs.sh

# ---------------------------------------------------------------------------
# Build-time model / dictionary download
# ---------------------------------------------------------------------------
# Assets are baked under /opt/geneformer-assets (NOT under /app) so Compose's
# `.:/app` bind mount cannot hide them. The entrypoint seeds into /app/models
# and /app/core/geneformer/dicts when those paths are empty/incomplete; host files win.
#
# DOWNLOAD_MODELS=default|minimal|all|none
# Optional build-arg HF_TOKEN if Hugging Face rate-limits anonymous access
# (not persisted as an image ENV). Build needs network: Google Drive, HF, BioMart.
ARG DOWNLOAD_MODELS=default
ARG HF_TOKEN=
ARG SKIP_ORTHOLOGS=0

ENV GENEFORMER_IMAGE_ASSETS=/opt/geneformer-assets

RUN mkdir -p /opt/geneformer-assets && \
    if [ "$DOWNLOAD_MODELS" = "none" ]; then \
      echo "DOWNLOAD_MODELS=none — skipping model download"; \
      GENEFORMER_ROOT=/opt/geneformer-assets DOWNLOAD_MODELS=none \
        bash /app/scripts/download_build_assets.sh; \
    else \
      echo "Downloading build assets (DOWNLOAD_MODELS=$DOWNLOAD_MODELS)..."; \
      GENEFORMER_ROOT=/opt/geneformer-assets \
      DOWNLOAD_MODELS="$DOWNLOAD_MODELS" \
      SKIP_ORTHOLOGS="$SKIP_ORTHOLOGS" \
      HF_TOKEN="$HF_TOKEN" \
      HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
        bash /app/scripts/download_build_assets.sh || { \
          echo "ERROR: build-time model/dictionary download failed." >&2; \
          echo "Check network access to Google Drive, Hugging Face, and Ensembl BioMart." >&2; \
          echo "Optional: docker compose build --build-arg HF_TOKEN=\$HF_TOKEN ..." >&2; \
          echo "Or rebuild with --build-arg DOWNLOAD_MODELS=none and download manually." >&2; \
          exit 1; \
        }; \
    fi

# Runtime uses /app paths; image bake lives under /opt (seeded by entrypoint)
ENV GENEFORMER_ROOT=/app \
    GENEFORMER_IMAGE_ASSETS=/opt/geneformer-assets

# Arch-aware OpenMP preload (aarch64 only) + model seed; see scripts/container-entrypoint.sh
ENTRYPOINT ["/app/scripts/container-entrypoint.sh"]

# Set the default command to bash
CMD ["bash"]
