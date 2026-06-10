# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
RUN uv sync --frozen --no-dev

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS runtime
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    TZ=Europe/Prague
WORKDIR /app
# tzdata + TZ make datetime.now() local to Prague (CEST), matching the legacy
# server's timezone so collected timestamps stay continuous with history.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 10001 bot && mkdir -p /data && chown bot:bot /data
COPY --from=builder --chown=bot:bot /app /app
USER bot
VOLUME ["/data"]
CMD ["python", "-m", "bot"]
