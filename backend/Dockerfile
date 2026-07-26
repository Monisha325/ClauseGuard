FROM python:3.12-slim

WORKDIR /app

# Optional: PEM contents of a corporate/interception root CA (e.g. an
# antivirus or proxy that TLS-inspects outbound HTTPS) to trust during
# build-time network calls below (pip install, and tiktoken's own fetch of
# its encoding data from a *different* host than PyPI). Leave unset on
# machines without such interception. Supply on build with:
#   docker compose build --build-arg PIP_CA_CERT="$(cat /path/to/ca.pem)"
ARG PIP_CA_CERT=""
# M31: tesseract-ocr (the real OCR binary) + tesseract-ocr-eng (English
# trained data -- NOT pulled in automatically by --no-install-recommends,
# confirmed directly: it's a "Recommends", not a "Depends", of the base
# tesseract-ocr package on Debian) installed alongside ca-certificates in
# this SAME apt-get layer. pytesseract (added to requirements.txt below)
# is only a thin Python wrapper around this real binary -- confirmed
# directly inside a running container that `pip install pytesseract`
# alone does NOT provide a working `tesseract` executable; this apt-get
# install is what actually does.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/* \
    && if [ -n "$PIP_CA_CERT" ]; then \
         printf '%s\n' "$PIP_CA_CERT" > /usr/local/share/ca-certificates/pip-build-ca.crt && \
         update-ca-certificates; \
       fi
# PIP_CERT covers pip itself; REQUESTS_CA_BUNDLE/SSL_CERT_FILE cover any
# other Python library using `requests`/stdlib ssl (e.g. tiktoken's
# encoding download) against the same trusted system CA store.
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

COPY requirements.txt .
# M21: torch (sentence-transformers' dependency) is installed FIRST, from
# PyTorch's own CPU-only wheel index, before the rest of requirements.txt.
# Without this, pip's default resolution against PyPI pulls torch's
# CUDA-bundled build (an extra multi-GB of unused nvidia_cuda*/nvidia_cudnn*
# packages) even though this app only ever runs the cross-encoder on CPU
# (confirmed empirically during this milestone's build -- a first attempt
# without this step started downloading nvidia_cudnn_cu13 etc. before being
# caught and fixed). Pinned to the same torch version sentence-transformers
# 5.6.0 itself resolves to, so the later `pip install -r requirements.txt`
# sees it already satisfied and does not reinstall/redownload it.
RUN pip install --no-cache-dir torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Pre-warm tiktoken's encoding data into the image at build time so the
# chunker never needs a runtime network call to fetch it (relevant both for
# offline/airgapped deployment and for this machine's AVG TLS interception).
ENV TIKTOKEN_CACHE_DIR=/app/.tiktoken_cache
RUN mkdir -p /app/.tiktoken_cache && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# M21: pre-warm the cross-encoder reranker model into the image at build
# time too, same reasoning as tiktoken above -- avoids a first-request
# latency spike AND a runtime network call through this machine's AVG TLS
# interception. Model name here must stay in sync with
# retrieval/reranker.py's MODEL_NAME constant by hand.
ENV HF_HOME=/app/.hf_cache
RUN mkdir -p /app/.hf_cache && python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')"

# CRITICAL, confirmed empirically (not assumed): huggingface_hub's default
# behavior makes LIVE network metadata-check calls (HEAD requests for
# config.json/tokenizer_config.json/etc.) on every CrossEncoder(...) load,
# even when the weights are already fully cached locally -- this measured
# ~62 SECONDS on the very first real request in this environment (AVG TLS
# interception adds real per-request overhead), a direct violation of the
# ~450ms p95 latency target. HF_HUB_OFFLINE=1 forces huggingface_hub to use
# ONLY the local cache populated by the RUN step above and never touch the
# network at runtime -- set here (AFTER the prewarm step, which still
# needs real network access to fetch the model the first time) so it only
# affects the running container, not the build.
ENV HF_HUB_OFFLINE=1

COPY backend/ .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
