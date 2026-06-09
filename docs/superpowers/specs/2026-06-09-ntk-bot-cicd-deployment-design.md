# ntk-bot — Professional CI/CD Deployment on the `azamat` VPS

**Date:** 2026-06-09
**Status:** Approved
**Repo:** `github.com/vsem-azamat/ntk-bot` (own repo — not merged into khromberry)

## Problem

`ntk-bot` is currently deployed by hand: `python3 -m bot` running inside a
`tmux` session on the legacy `poryadok` Contabo VPS, from an old 2024 `venv`
checkout in `~/projects/NTK-bot`. There is no CI, no reproducible build, no
automated deploy, and the valuable occupancy time-series lives in a flat file
that is not backed up. We want a professional, reproducible CI/CD pipeline that
deploys the modern `ntk-bot` codebase to the `azamat` VPS, mirroring the proven
pattern already used by the `khromberry` project (used purely as a reference
template — the two products stay in separate repos).

## Goals

- One-command-free deploys: push to `master` → CI gates → image to GHCR →
  Docker Compose stack restarted on the `azamat` VPS.
- Reproducible container build (no host `venv`, no manual `tmux`).
- Migrate all live data and credentials off the legacy `poryadok` VPS.
- Move the occupancy time-series from a flat file into **SQLite** on a
  persistent Docker volume.
- Quality gates in CI: `ruff` format + lint, `ty` type-check, `pytest`,
  Docker build.

## Non-Goals

- No web UI, no inbound HTTP, no subdomain (the bot is Telegram long-polling).
- No Postgres migration (SQLite is sufficient and was chosen).
- No rewrite of bot features; this is a lift-shift + persistence change.
- Not migrating the 670 MB RandomForest `.pkl` — models are retrained on-host.

## Topology

| | Legacy (decommission) | Target |
|---|---|---|
| Host | `poryadok` Contabo VPS (`vmi1519048`) | `azamat` VPS `46.225.117.31`, user `azamat` |
| Run | `python3 -m bot` in `tmux` session `ntk` | Docker Compose stack |
| Code | `~/projects/NTK-bot` (old venv, `NTK-bot.git`) | GHCR image from `ntk-bot.git` |
| Deploy dir | n/a | `/home/azamat/deploy/ntk-bot` |
| Data | flat `ntk_data.txt`, `*.pkl`, `.instructions`, `.env` | SQLite + models on `ntk_data:/data` volume |

**Hard constraint:** a Telegram bot token can be long-polled by **exactly one**
process at a time (a second `getUpdates` consumer causes HTTP 409). Therefore
cutover is **stop-legacy-then-start-new**, accepting a brief downtime window.

## Persistence: flat file → SQLite

The occupancy series currently lives in `ntk_data.txt` with lines of the form:

```
2023-09-24 12:10 - 118
```

### Data module

Introduce a small persistence module (`bot/db.py`) exposing a focused API so
the four current touch points depend on an interface, not on file I/O:

- `init_db() -> None` — create schema if absent (called on startup).
- `insert_occupancy(ts: datetime, people: int) -> None` — idempotent upsert.
- `get_occupancy(start: datetime, end: datetime) -> list[tuple[datetime, int]]`
  — range read for plotting.
- `get_all_occupancy() -> list[tuple[datetime, int]]` — full read for ML
  training.
- `export_occupancy_text() -> bytes` — reproduce the legacy `ntk_data.txt`
  format for the `/data` dump command.

### Schema

```sql
CREATE TABLE IF NOT EXISTS occupancy (
    ts     TEXT PRIMARY KEY,   -- ISO 'YYYY-MM-DD HH:MM'
    people INTEGER NOT NULL
);
```

`ts` as PRIMARY KEY makes repeated collection of the same minute idempotent
(`INSERT ... ON CONFLICT(ts) DO UPDATE`).

### Touch points to swap

| File | Current | After |
|---|---|---|
| `apps/schedule_functions.py` | `open(NTK_DATA_PATH,"a")` append | `db.insert_occupancy(...)` |
| `apps/predictModels.py` | `open(NTK_DATA_PATH)` read-all | `db.get_all_occupancy()` |
| `apps/plot_functions.py` | `open(NTK_DATA_PATH)` + date filter | `db.get_occupancy(start, end)` |
| `bot/handlers/data.py` | sends raw `ntk_data.txt` | `db.export_occupancy_text()` → `BufferedInputFile` |

### Models / `DATA_DIR`

Add a `DATA_DIR` config (default `.` for local dev, `/data` in the container).
SQLite path, `.instructions`, and `model_*.pkl` all resolve under `DATA_DIR`.
`predictModels.py` dump and `plot_functions.py` load use `DATA_DIR`. Models are
**retrained on the new host** via the existing `/learn` flow after migration.

### Migration script

`scripts/import_legacy_ntk_data.py`:
- Reads a legacy `ntk_data.txt` path (arg).
- Parses each `YYYY-MM-DD HH:MM - N` line; skips malformed lines (logs count).
- Bulk-inserts into the target SQLite DB (idempotent).
- Prints rows imported / skipped / final row count.
- Runnable inside the container against the `/data` volume.

## Repository additions (mirrors khromberry, adapted for Python/uv)

### `infra/Dockerfile`

Multi-stage, `uv`-based:
- Builder: `python:3.11-slim` + `uv`, `uv sync --frozen --no-dev` into a venv.
- Runtime: `python:3.11-slim`, copy the venv + source, set
  `MPLBACKEND=Agg` (headless matplotlib), non-root user, `CMD ["python","-m","bot"]`.
- Build deps for scipy/scikit-learn/numpy resolved via wheels (slim + build
  toolchain only if needed).

### `infra/docker-compose.prod.yml`

Single service `bot`:
- `image: ${NTK_BOT_IMAGE:?...}`, `restart: unless-stopped`.
- `env_file: .env`.
- `volumes: [ ntk_data:/data ]`, `environment: DATA_DIR=/data`.
- **Healthcheck** on a heartbeat file: the bot touches `/data/heartbeat` each
  poll loop; healthcheck fails if the file is older than ~5 min, so a hung bot
  auto-restarts. No `ports:` (no inbound).
- Log rotation (`max-size: 10m`, `max-file: 3`).
- Named volume `ntk_data`.

### `infra/.env.production.example`

```
NTK_BOT_IMAGE=ghcr.io/vsem-azamat/ntk-bot:latest
BOT_TOKEN=
SUPER_ADMINS=
DELTA_TIME=20
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openai/gpt-4o
ANSWER_PROBABILITY=0.025
DATA_DIR=/data
```

### `infra/deploy.sh`

`docker compose --env-file .env -f infra/docker-compose.prod.yml pull && up -d
--remove-orphans`, then poll `docker compose ps` / healthcheck until the `bot`
service is healthy; dump recent logs and exit non-zero on failure.

### `.github/workflows/ci.yml`

On PR + push to `master` + `workflow_dispatch`:
`uv sync` → `uv run ruff format --check .` → `uv run ruff check .` →
`uv run ty check` → `uv run pytest` → validate compose config →
`docker build -f infra/Dockerfile`.

### `.github/workflows/deploy.yml`

On CI success on `master` (`workflow_run`) + `workflow_dispatch`:
build & push `ghcr.io/vsem-azamat/ntk-bot` (sha + `latest`) → SSH to azamat →
rsync `infra/` → render `/home/azamat/deploy/ntk-bot/.env` from secrets →
run `infra/deploy.sh`. Deploy-dir and target guards as in khromberry.

### GitHub secrets

Reuse the same-VPS SSH secrets: `DEMO_SSH_HOST`, `DEMO_SSH_USER`,
`DEMO_SSH_KEY`, `DEMO_SSH_PORT`. Add bot secrets: `BOT_TOKEN`, `SUPER_ADMINS`,
`OPENROUTER_API_KEY` (optional), `OPENROUTER_MODEL` (optional). Deploy dir
`/home/azamat/deploy/ntk-bot`.

## Cutover sequence

1. Land the SQLite refactor + `infra/` + workflows on `feature/cicd-deployment`;
   CI green; merge to `master`.
2. Provision GitHub secrets; create `/home/azamat/deploy/ntk-bot`; ensure the
   azamat host is logged in to GHCR.
3. **Stop the legacy bot** — kill the `python3 -m bot` process in tmux `ntk` on
   `poryadok`. This freezes `ntk_data.txt`.
4. `rsync` the final `ntk_data.txt` + `.instructions` from poryadok to the
   azamat deploy host; import `ntk_data.txt` into `/data/ntk.sqlite` via the
   migration script (in-container); place `.instructions` in `/data`.
5. Start (or let the deploy start) the new stack; retrain models on-host via
   `/learn`; verify `/ntk`, `/graph`, `/weather`, and that collection appends
   new rows to SQLite.
6. Decommission legacy: archive `ntk_data.txt`, remove the tmux session, mark
   `NTK-bot` repo/dir as retired.

## Credential mapping

| Legacy `.env` | New | Note |
|---|---|---|
| `BOT_TOKEN` | `BOT_TOKEN` | same token; ensure legacy stopped first |
| `SUPER_ADMINS` | `SUPER_ADMINS` | unchanged |
| `DELTA_TIME` | `DELTA_TIME` | unchanged |
| `OPENAI_API_KEY` | `OPENROUTER_API_KEY` | **different provider** — old key will not work with OpenRouter |

GPT random replies are optional and degrade gracefully when
`OPENROUTER_API_KEY` is empty. Ship with GPT disabled until a real OpenRouter
key is provided.

## Risks & mitigations

- **Double-polling 409** → strict stop-legacy-before-start-new ordering.
- **Data loss of final rows** → export `ntk_data.txt` only *after* stopping the
  legacy bot.
- **Image size** (scipy/sklearn/matplotlib) → slim base + wheels; acceptable
  for a single worker. Revisit only if build times hurt.
- **Lost models** → intentional; retrain on-host from the migrated series.

## Testing

- Unit tests for `bot/db.py` (insert/upsert/range/all/export round-trips on a
  temp SQLite file).
- Unit test for the migration parser (valid + malformed lines).
- Keep existing `tests/` green through the refactor.
- CI Docker build proves the runtime image assembles.
```
