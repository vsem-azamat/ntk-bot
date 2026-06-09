#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"
COMPOSE_FILE="$ROOT_DIR/infra/docker-compose.prod.yml"

command -v docker >/dev/null 2>&1 || { echo "docker is required on the host" >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "docker compose plugin is required" >&2; exit 1; }
[ -f "$ENV_FILE" ] || { echo "Missing env file: $ENV_FILE" >&2; exit 1; }

dc() { docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "$@"; }

echo "[deploy] validating compose configuration..."
dc config >/dev/null

echo "[deploy] pulling bot image..."
dc pull

echo "[deploy] starting stack..."
dc up -d --remove-orphans

echo "[deploy] waiting for the bot container to report healthy..."
cid="$(dc ps -q bot)"
if [ -z "$cid" ]; then
  echo "[deploy] bot container did not start" >&2
  dc logs --tail=120 bot >&2
  exit 1
fi
for attempt in $(seq 1 30); do
  status="$(docker inspect -f '{{.State.Health.Status}}' "$cid" 2>/dev/null || true)"
  if [ "$status" = "healthy" ]; then
    echo "[deploy] bot is healthy"
    dc ps
    exit 0
  fi
  sleep 3
  echo "[deploy] health retry ${attempt}/30 (status=${status:-unknown})"
done

echo "[deploy] health check failed; recent logs:" >&2
dc logs --tail=120 bot >&2
exit 1
