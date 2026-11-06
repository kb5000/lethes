# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.12.10 AS uv
FROM python:3.12-slim-bookworm
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/opt/lethes-venv \
    UV_LINK_MODE=copy \
    PATH=/opt/lethes-venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LITELLM_LOCAL_MODEL_COST_MAP=True \
    LETHES_STATE_PATH=/data/lethes.sqlite3 \
    LETHES_CONFIG_PATH=/app/config/lethes.local.json \
    HF_HOME=/data/hf-cache \
    TIKTOKEN_CACHE_DIR=/data/tiktoken
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --locked --no-dev --extra server --extra litellm \
    && useradd --create-home --uid 10001 --user-group lethes \
    && mkdir -p /data /app/config \
    && chown lethes:lethes /data
USER lethes
EXPOSE 8000
CMD ["uvicorn", "lethes.service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
