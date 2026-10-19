# syntax=docker/dockerfile:1
# Multi-arch node image (linux/amd64 and linux/arm64 / Raspberry Pi 4 on 64-bit OS).
#   docker buildx build --platform linux/amd64,linux/arm64 -f deploy/Dockerfile -t edgecluster:latest .
# Only the runtime dependencies are installed: no torch/transformers on the edge.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    EDGE_DATA_DIR=/data

WORKDIR /app
COPY requirements/runtime.txt requirements/runtime.txt
RUN pip install -r requirements/runtime.txt

COPY pyproject.toml README.md ./
COPY edgecluster edgecluster
RUN pip install --no-deps . \
    && useradd --system --uid 1000 edge \
    && mkdir -p /data /models /config \
    && chown edge /data

USER edge
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
ENTRYPOINT ["python", "-m", "edgecluster.serving"]
CMD ["--config", "/config/cluster.yaml"]
