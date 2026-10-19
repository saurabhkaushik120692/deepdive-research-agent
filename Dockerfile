# syntax=docker/dockerfile:1
# --- build: resolve dependencies with uv into /app/.venv -------------------------
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# --- runtime: slim image, non-root, no secrets baked in ---------------------------
FROM python:3.11-slim-bookworm
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY app ./app
COPY eval ./eval
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/data \
    DATABASE_URL=sqlite:////app/data/deepdive.db
RUN mkdir -p /app/data && chown -R app:app /app/data
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
