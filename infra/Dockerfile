# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim AS builder
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /app
# Dependencies only: this stage (and the layer copied from it) changes only when
# pyproject.toml / uv.lock change. The app runs from source via PYTHONPATH, so
# the project itself is not installed into the venv.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim AS runtime
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
# Two layers on purpose: the ~860 MB virtualenv is reused by the host across
# deploys, so a code-only change pulls just the small source layer.
COPY --from=builder --chown=bot:bot /app/.venv /app/.venv
COPY --chown=bot:bot . /app
USER bot
VOLUME ["/data"]
CMD ["python", "-m", "bot"]
