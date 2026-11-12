FROM ghcr.io/astral-sh/uv:0.13.7 AS uv
FROM python:3.12-slim

COPY --from=uv /uv /uvx /bin/
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/home/app/.cache/huggingface \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /home/app/.cache/huggingface \
    && chown -R app:app /home/app/.cache

WORKDIR /app
# Shared CPU/GPU image by default. Local CPU-only builds can override this argument.
ARG BGE_EXTRA=cuda
COPY pyproject.toml uv.lock .python-version ./
# Do not retain uv's package cache in the image (CUDA wheels are large).
RUN uv sync --frozen --no-dev --extra "${BGE_EXTRA}" --no-cache \
    && .venv/bin/python -c "from FlagEmbedding import BGEM3FlagModel"
COPY app ./app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=4)"
# One process = one model copy. Do not increase workers on a single GPU.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
