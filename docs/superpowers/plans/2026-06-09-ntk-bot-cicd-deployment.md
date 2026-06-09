# ntk-bot CI/CD Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hand-run `tmux` deployment of ntk-bot on the legacy `poryadok` VPS with a reproducible GitHub Actions → GHCR → Docker Compose pipeline on the `azamat` VPS, moving occupancy data from a flat file into SQLite on a persistent volume.

**Architecture:** The bot stays a single Telegram long-polling worker (no inbound port). Persistence moves to SQLite under a configurable `DATA_DIR` (`/data` volume in the container). CI runs ruff/ty/pytest/docker-build; a deploy workflow builds and pushes a `uv`-based image to GHCR, rsyncs `infra/` to the VPS, renders `.env` from secrets, and restarts the Compose stack. A one-off script migrates the legacy `ntk_data.txt` into SQLite.

**Tech Stack:** Python 3.11, `uv`, aiogram, scikit-learn, SQLite (stdlib `sqlite3`), Docker, Docker Compose, GitHub Actions, GHCR.

**Reference spec:** `docs/superpowers/specs/2026-06-09-ntk-bot-cicd-deployment-design.md`

**Key design decision — least-invasive data layer:** the existing code parses
data rows as `"YYYY-MM-DD HH:MM - N"` strings via `parse_row_datetime` and
`predictModels.remove_zero_values`. We KEEP that format and those functions
(so all current tests stay green) and only change the *source* of rows from a
flat file to SQLite. `bot/db.py` exposes rows in the exact legacy string format.

---

## File Structure

| Path | Responsibility | Action |
|---|---|---|
| `config.py` | adds `DATA_DIR`, `DB_PATH`; `.instructions` under `DATA_DIR`; drops flat-file constants | Modify |
| `bot/db.py` | SQLite persistence (init, insert, read rows, export) | Create |
| `apps/schedule_functions.py` | write occupancy to SQLite + heartbeat | Modify |
| `apps/predictModels.py` | read training rows from SQLite; model path under `DATA_DIR` | Modify |
| `apps/plot_functions.py` | read plot rows from SQLite | Modify |
| `bot/handlers/data.py` | `/data` dump from SQLite | Modify |
| `bot/bot.py` | init DB on startup | Modify |
| `scripts/import_legacy_ntk_data.py` | one-off legacy `ntk_data.txt` → SQLite | Create |
| `tests/bot/test_db.py` | unit tests for `bot/db.py` | Create |
| `tests/scripts/test_import_legacy.py` | unit tests for the migration parser | Create |
| `.gitignore` | ignore `*.sqlite`, `heartbeat` | Modify |
| `infra/Dockerfile` | uv-based multi-stage image | Create |
| `infra/docker-compose.prod.yml` | single `bot` service + volume + healthcheck | Create |
| `infra/.env.production.example` | env template | Create |
| `infra/deploy.sh` | remote pull + up + health wait | Create |
| `.dockerignore` | build context trim | Create |
| `.github/workflows/ci.yml` | lint/type/test/build gates | Create |
| `.github/workflows/deploy.yml` | build/push/deploy | Create |
| `docs/deployment/prod.md` | runbook + cutover checklist | Create |
| `README.md` | deployment + SQLite notes | Modify |

---

## Part A — SQLite persistence (code, TDD)

### Task 1: Add `DATA_DIR` / `DB_PATH` configuration

**Files:**
- Modify: `config.py`

- [ ] **Step 1: Edit `config.py`**

Replace the top-of-file constants and the `Config.FILES` section. The new
top of the file (lines 1–7) becomes:

```python
import os
from pathlib import Path

from decouple import config

# Directory that holds all mutable runtime state (SQLite DB, trained models,
# the GPT instructions file, the liveness heartbeat). "." for local dev,
# "/data" (a Docker volume) in production.
DATA_DIR: str = config("DATA_DIR", cast=str, default=".")

BAD_WORDS_PATH = "bad_words.txt"
INSTRUCTIONS_PATH = str(Path(DATA_DIR) / ".instructions")
DB_PATH = str(Path(DATA_DIR) / "ntk.sqlite")
HEARTBEAT_PATH = str(Path(DATA_DIR) / "heartbeat")
```

In class `Config`, replace the `# FILES` block so it reads:

```python
    # >>>>>>>>>> FILES <<<<<<<<<< #
    BAD_WORDS: list[str] = _load_lines(BAD_WORDS_PATH)
    DATA_DIR: str = DATA_DIR
    DB_PATH: str = DB_PATH
    HEARTBEAT_PATH: str = HEARTBEAT_PATH
```

Delete the now-unused `NTK_DATA_PATH = "ntk_data.txt"` module constant, the
`NTK_DATA_PATH: str = NTK_DATA_PATH` class attribute, and the
`ensure_data_file()` function at the bottom (DB creation replaces it).

- [ ] **Step 2: Verify config imports cleanly**

Run: `BOT_TOKEN=x uv run python -c "from config import cnfg; print(cnfg.DB_PATH, cnfg.DATA_DIR)"`
Expected: prints `ntk.sqlite .`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "feat(config): add DATA_DIR/DB_PATH, drop flat-file constants"
```

---

### Task 2: Create the `bot/db.py` SQLite module

**Files:**
- Create: `bot/db.py`
- Test: `tests/bot/test_db.py`

- [ ] **Step 1: Write the failing test**

Create `tests/bot/test_db.py`:

```python
from datetime import datetime

import pytest

from bot import db


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    path = tmp_path / "ntk.sqlite"
    monkeypatch.setattr(db.cnfg, "DB_PATH", str(path))
    db.init_db()
    return path


def test_insert_and_iter_rows(tmp_db):
    db.insert_occupancy(datetime(2024, 3, 1, 9, 30), 42)
    db.insert_occupancy(datetime(2024, 3, 1, 9, 50), 7)
    assert db.iter_rows() == ["2024-03-01 09:30 - 42", "2024-03-01 09:50 - 7"]


def test_insert_is_idempotent_upsert(tmp_db):
    db.insert_occupancy(datetime(2024, 3, 1, 9, 30), 42)
    db.insert_occupancy(datetime(2024, 3, 1, 9, 30), 99)
    assert db.iter_rows() == ["2024-03-01 09:30 - 99"]


def test_rows_are_ordered_by_time(tmp_db):
    db.insert_occupancy(datetime(2024, 3, 1, 9, 50), 7)
    db.insert_occupancy(datetime(2024, 3, 1, 9, 30), 42)
    assert db.iter_rows() == ["2024-03-01 09:30 - 42", "2024-03-01 09:50 - 7"]


def test_export_text_round_trips(tmp_db):
    db.insert_occupancy(datetime(2024, 3, 1, 9, 30), 42)
    assert db.export_text() == b"2024-03-01 09:30 - 42\n"


def test_export_text_empty_db(tmp_db):
    assert db.export_text() == b""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/bot/test_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bot.db'`

- [ ] **Step 3: Create `bot/db.py`**

```python
"""SQLite persistence for NTK occupancy samples.

Rows are exposed in the legacy ``"YYYY-MM-DD HH:MM - N"`` text format so the
existing parsing/ML/plotting code keeps working unchanged.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

from config import cnfg

_TS_FORMAT = "%Y-%m-%d %H:%M"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(cnfg.DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create the data directory and occupancy table if they do not exist."""
    Path(cnfg.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS occupancy ("
            " ts TEXT PRIMARY KEY,"
            " people INTEGER NOT NULL)"
        )


def insert_occupancy(ts: datetime, people: int) -> None:
    """Insert one sample, overwriting any existing row for the same minute."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO occupancy (ts, people) VALUES (?, ?)"
            " ON CONFLICT(ts) DO UPDATE SET people=excluded.people",
            (ts.strftime(_TS_FORMAT), people),
        )


def iter_rows() -> list[str]:
    """Return all samples as legacy-format rows, oldest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT ts, people FROM occupancy ORDER BY ts"
        ).fetchall()
    return [f"{ts} - {people}" for ts, people in rows]


def export_text() -> bytes:
    """Render the whole series as a downloadable ``ntk_data.txt`` blob."""
    rows = iter_rows()
    if not rows:
        return b""
    return ("\n".join(rows) + "\n").encode("utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/bot/test_db.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add bot/db.py tests/bot/test_db.py
git commit -m "feat(db): add SQLite occupancy store with legacy-format rows"
```

---

### Task 3: Write occupancy to SQLite + liveness heartbeat

**Files:**
- Modify: `apps/schedule_functions.py`

- [ ] **Step 1: Replace `apps/schedule_functions.py`**

```python
import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path

from apps.collect_time import generate_time_list
from apps.parse_functions import get_ntk_quantity
from bot import db
from config import cnfg

logger = logging.getLogger(__name__)


def _touch_heartbeat() -> None:
    """Update the liveness file the container healthcheck inspects."""
    try:
        Path(cnfg.HEARTBEAT_PATH).touch()
    except OSError:
        logger.exception("Failed to update heartbeat file")


async def receive_ntk_data(delta_minutes: int = 20) -> None:
    """Collect occupancy data from the NTK website every ``delta_minutes``."""
    time_list = await generate_time_list(delta_minutes=delta_minutes)

    while True:
        _touch_heartbeat()
        current_time = datetime.now().strftime("%H:%M")
        if current_time in time_list:
            try:
                quantity_ntk = await get_ntk_quantity()
                db.insert_occupancy(datetime.now(), quantity_ntk)
            except Exception:
                logger.exception("Failed to collect NTK occupancy data")
            await asyncio.sleep(delta_minutes * 60 - 60)
        else:
            await asyncio.sleep(1)
```

(Note: `db.insert_occupancy` stores `datetime.now()`; seconds are dropped by
the `%Y-%m-%d %H:%M` format inside `db`, matching the old minute granularity.
The `time` import is used by the heartbeat-adjacent logic in later tasks; keep
it.) If `ruff` flags `time` as unused, remove the `import time` line.

- [ ] **Step 2: Run the suite to confirm nothing broke**

Run: `uv run pytest -v`
Expected: PASS (existing tests unaffected)

- [ ] **Step 3: Lint**

Run: `uv run ruff check apps/schedule_functions.py`
Expected: no errors (remove `import time` if reported unused)

- [ ] **Step 4: Commit**

```bash
git add apps/schedule_functions.py
git commit -m "feat(collect): persist occupancy to SQLite and write heartbeat"
```

---

### Task 4: Read training data from SQLite; model path under `DATA_DIR`

**Files:**
- Modify: `apps/predictModels.py`

- [ ] **Step 1: Update `model_filename` and `learn_models`**

At the top of `apps/predictModels.py`, add the import:

```python
from pathlib import Path
```

Replace `model_filename` (lines 16–18) with:

```python
def model_filename(model_name: str) -> str:
    """Return the on-disk path for a model given its class name."""
    return str(Path(cnfg.DATA_DIR) / f"model_{model_name}.pkl")
```

(`Path(".") / "model_X.pkl"` normalises to `model_X.pkl`, so the existing
`test_model_filename_has_no_parens` test still passes; in production
`DATA_DIR=/data` yields `/data/model_X.pkl`.)

Replace `learn_models` (lines 68–76) with:

```python
    async def learn_models(self) -> None:
        data = db.iter_rows()
        if len(data) > 10:
            await self.perform_regression(data, RandomForestRegressor())
            await self.perform_regression(data, GradientBoostingRegressor())
```

Add the import near the other imports:

```python
from bot import db
```

Remove the now-unused `from config import cnfg`? No — `cnfg` is still used by
`model_filename`. Keep it.

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/apps/test_predict_models.py -v`
Expected: PASS (model_filename, parse_row_datetime, extract_features, remove_zero_values all green)

- [ ] **Step 3: Commit**

```bash
git add apps/predictModels.py
git commit -m "feat(ml): train from SQLite and store models under DATA_DIR"
```

---

### Task 5: Read plot data from SQLite

**Files:**
- Modify: `apps/plot_functions.py`

- [ ] **Step 1: Update `get_ntk_data`**

Add the import near the top of `apps/plot_functions.py`:

```python
from bot import db
```

Replace the body of `get_ntk_data` (lines 30–46) with:

```python
    async def get_ntk_data(
        self, start_datetime: datetime, end_datetime: datetime
    ) -> tuple[list[datetime], list[int]]:
        """Get occupancy data from SQLite within the given datetime range."""
        datetimes: list[datetime] = []
        quantities: list[int] = []
        data = await predictModels.remove_zero_values(db.iter_rows())
        for row in data:
            row_datetime = parse_row_datetime(row)
            if start_datetime <= row_datetime <= end_datetime:
                quantities.append(int(row.split(" - ")[1].strip()))
                datetimes.append(row_datetime)
        return datetimes, quantities
```

`cnfg` may now be unused in this file — if `ruff` reports
`from config import cnfg` as unused, remove that import line.

- [ ] **Step 2: Run tests + lint**

Run: `uv run pytest -v && uv run ruff check apps/plot_functions.py`
Expected: PASS, no lint errors

- [ ] **Step 3: Commit**

```bash
git add apps/plot_functions.py
git commit -m "feat(plot): source occupancy series from SQLite"
```

---

### Task 6: `/data` dump command from SQLite

**Files:**
- Modify: `bot/handlers/data.py`

- [ ] **Step 1: Update `send_data`**

In `bot/handlers/data.py`, add the import:

```python
from bot import db
```

Replace `send_data` (lines 61–66) with:

```python
@router.message(Command("data"), SuperAdmins())
async def send_data(msg: types.Message, bot: Bot):
    """Send the occupancy series exported from SQLite."""
    input_file = types.BufferedInputFile(
        file=db.export_text(), filename="ntk_data.txt"
    )
    await bot.send_document(msg.chat.id, input_file)
```

If `from config import cnfg` is now unused in this file, remove it.

- [ ] **Step 2: Lint + tests**

Run: `uv run ruff check bot/handlers/data.py && uv run pytest -v`
Expected: no lint errors, tests PASS

- [ ] **Step 3: Commit**

```bash
git add bot/handlers/data.py
git commit -m "feat(handlers): export /data dump from SQLite"
```

---

### Task 7: Initialise the DB on startup

**Files:**
- Modify: `bot/bot.py`

- [ ] **Step 1: Update `bot/bot.py`**

Change the import line 8 from:

```python
from config import INSTRUCTIONS_PATH, cnfg, ensure_data_file
```

to:

```python
from bot import db
from config import INSTRUCTIONS_PATH, cnfg
```

In `on_startup`, replace `ensure_data_file()` (line 17) with:

```python
    db.init_db()
```

- [ ] **Step 2: Smoke-check import**

Run: `BOT_TOKEN=x uv run python -c "import bot.bot"`
Expected: no error

- [ ] **Step 3: Run full suite**

Run: `uv run pytest -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add bot/bot.py
git commit -m "feat(startup): initialise SQLite DB on boot"
```

---

### Task 8: Legacy data migration script

**Files:**
- Create: `scripts/import_legacy_ntk_data.py`
- Test: `tests/scripts/test_import_legacy.py`

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_import_legacy.py`:

```python
from datetime import datetime

import pytest

from bot import db
from scripts.import_legacy_ntk_data import import_file


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db.cnfg, "DB_PATH", str(tmp_path / "ntk.sqlite"))
    db.init_db()
    return tmp_path


def test_import_valid_and_malformed(tmp_db):
    legacy = tmp_db / "ntk_data.txt"
    legacy.write_text(
        "2023-09-24 12:10 - 118\n"
        "2023-09-24 12:20 - 120\n"
        "\n"  # blank
        "garbage without separator\n"  # malformed
        "2023-09-24 12:30 - 127\n"
    )
    imported, skipped = import_file(str(legacy))
    assert (imported, skipped) == (3, 2)
    assert db.iter_rows() == [
        "2023-09-24 12:10 - 118",
        "2023-09-24 12:20 - 120",
        "2023-09-24 12:30 - 127",
    ]


def test_import_is_idempotent(tmp_db):
    legacy = tmp_db / "ntk_data.txt"
    legacy.write_text("2023-09-24 12:10 - 118\n")
    import_file(str(legacy))
    import_file(str(legacy))
    assert db.iter_rows() == ["2023-09-24 12:10 - 118"]
```

Create empty `tests/scripts/__init__.py` and `scripts/__init__.py` so the
imports resolve (pytest `pythonpath = "."` is already set in `pyproject.toml`).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scripts/test_import_legacy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.import_legacy_ntk_data'`

- [ ] **Step 3: Create `scripts/import_legacy_ntk_data.py`**

```python
"""One-off importer: legacy flat ``ntk_data.txt`` -> SQLite.

Usage (inside the container against the /data volume):
    python scripts/import_legacy_ntk_data.py /data/legacy.txt
"""

import sys
from datetime import datetime

from bot import db

_TS_FORMAT = "%Y-%m-%d %H:%M"


def import_file(path: str) -> tuple[int, int]:
    """Import every valid ``'<ts> - <count>'`` line. Returns (imported, skipped)."""
    db.init_db()
    imported = 0
    skipped = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = line.strip()
            if not row:
                skipped += 1
                continue
            try:
                ts_part, count_part = row.split(" - ", 1)
                ts = datetime.strptime(ts_part.strip(), _TS_FORMAT)
                people = int(count_part.strip())
            except (ValueError, IndexError):
                skipped += 1
                continue
            db.insert_occupancy(ts, people)
            imported += 1
    return imported, skipped


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python scripts/import_legacy_ntk_data.py <ntk_data.txt>")
        raise SystemExit(2)
    imported, skipped = import_file(sys.argv[1])
    total = len(db.iter_rows())
    print(f"imported={imported} skipped={skipped} total_rows_in_db={total}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/scripts/test_import_legacy.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/import_legacy_ntk_data.py scripts/__init__.py tests/scripts/
git commit -m "feat(migrate): legacy ntk_data.txt -> SQLite importer"
```

---

### Task 9: Update `.gitignore`

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Edit `.gitignore`**

Replace `ntk_data.txt` with the SQLite + heartbeat artefacts. Final file:

```gitignore
.env
.vscode/
.idea/
*__pycache__/
*.py[cod]
logs/
*.log
*.sqlite
*.sqlite-wal
*.sqlite-shm
heartbeat
venv/
.venv/
.instructions
*.pkl
```

- [ ] **Step 2: Verify nothing tracked is now ignored unexpectedly**

Run: `git status --porcelain`
Expected: clean (no tracked DB/heartbeat files existed)

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: ignore SQLite and heartbeat artefacts"
```

---

### Task 10: Full local verification gate

- [ ] **Step 1: Run every CI gate locally**

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest -v
```
Expected: all four pass. Fix any formatting with `uv run ruff format .` and
recommit before proceeding.

- [ ] **Step 2: Manual end-to-end smoke (optional, requires a test BOT_TOKEN)**

```bash
DATA_DIR=./tmp-data BOT_TOKEN=<test-token> uv run python -m bot
```
Expected: logs `OPENROUTER_API_KEY is not set; GPT replies are disabled`,
creates `tmp-data/ntk.sqlite`, starts polling. Ctrl-C to stop;
`rm -rf tmp-data`.

---

## Part B — Container & infra

### Task 11: `.dockerignore` and `infra/Dockerfile`

**Files:**
- Create: `.dockerignore`
- Create: `infra/Dockerfile`

- [ ] **Step 1: Create `.dockerignore`**

```dockerignore
.git
.github
.venv
venv
**/__pycache__
*.pyc
*.log
logs
*.pkl
*.sqlite
*.sqlite-*
heartbeat
.env
.env.*
!infra/.env.production.example
example_images
docs
tests
*.md
```

(Note: `icons/` is intentionally NOT ignored — the weather plot overlays
`icons/*.png` at runtime. `README.md` is excluded by `*.md`; that is fine
because `pyproject.toml` reads it only at build metadata time, which is not
required for `uv sync`. If `uv sync` errors about a missing README, add
`!README.md` to this file.)

- [ ] **Step 2: Create `infra/Dockerfile`**

```dockerfile
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
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data
WORKDIR /app
RUN useradd -m -u 10001 bot && mkdir -p /data && chown bot:bot /data
COPY --from=builder --chown=bot:bot /app /app
USER bot
VOLUME ["/data"]
CMD ["python", "-m", "bot"]
```

- [ ] **Step 3: Build the image locally**

Run: `docker build -f infra/Dockerfile -t ntk-bot:dev .`
Expected: build succeeds (scipy/scikit-learn/numpy/matplotlib install from
manylinux wheels — no compiler needed). If `uv sync` fails on the missing
README, add `!README.md` to `.dockerignore` and rebuild.

- [ ] **Step 4: Smoke-run the image (no token → expect token error, proves wiring)**

Run: `docker run --rm -e BOT_TOKEN=x ntk-bot:dev python -c "import bot.bot, bot.db; print('ok')"`
Expected: prints `ok`

- [ ] **Step 5: Commit**

```bash
git add .dockerignore infra/Dockerfile
git commit -m "feat(infra): uv-based Dockerfile and dockerignore"
```

---

### Task 12: `infra/docker-compose.prod.yml`

**Files:**
- Create: `infra/docker-compose.prod.yml`

- [ ] **Step 1: Create the compose file**

```yaml
name: ntk-bot

services:
  bot:
    image: ${NTK_BOT_IMAGE:?NTK_BOT_IMAGE is required}
    restart: unless-stopped
    env_file: ../.env
    environment:
      DATA_DIR: /data
    volumes:
      - ntk_data:/data
    healthcheck:
      test:
        - CMD
        - python
        - -c
        - >-
          import os,sys,time;
          p='/data/heartbeat';
          sys.exit(0 if os.path.exists(p) and time.time()-os.path.getmtime(p) < 180 else 1)
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 30s
    logging:
      options:
        max-size: "10m"
        max-file: "3"

volumes:
  ntk_data:
```

(`env_file: ../.env` resolves relative to the compose file in `infra/`, i.e.
`/home/azamat/deploy/ntk-bot/.env`. `DATA_DIR` is forced to `/data` regardless
of the `.env` value so the volume mount and config always agree.)

- [ ] **Step 2: Validate compose config**

```bash
cat > /tmp/ntk.env <<'EOF'
NTK_BOT_IMAGE=ghcr.io/vsem-azamat/ntk-bot:ci
BOT_TOKEN=ci-token
SUPER_ADMINS=
DELTA_TIME=20
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openai/gpt-4o
ANSWER_PROBABILITY=0.025
DATA_DIR=/data
EOF
docker compose --env-file /tmp/ntk.env -f infra/docker-compose.prod.yml config >/dev/null && echo OK
```
Expected: prints `OK`

- [ ] **Step 3: Commit**

```bash
git add infra/docker-compose.prod.yml
git commit -m "feat(infra): production compose with volume and heartbeat healthcheck"
```

---

### Task 13: `infra/.env.production.example` and `infra/deploy.sh`

**Files:**
- Create: `infra/.env.production.example`
- Create: `infra/deploy.sh`

- [ ] **Step 1: Create `infra/.env.production.example`**

```dotenv
# Image published by the deploy workflow.
NTK_BOT_IMAGE=ghcr.io/vsem-azamat/ntk-bot:latest

# Telegram bot token (the SAME token the legacy bot used; ensure the legacy
# process is stopped before this container starts — one poller per token).
BOT_TOKEN=

# Comma-separated Telegram user IDs allowed to run admin commands.
SUPER_ADMINS=

# Minutes between occupancy samples.
DELTA_TIME=20

# Optional OpenRouter credentials for random GPT replies. Leave empty to keep
# GPT replies disabled (the bot degrades gracefully).
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openai/gpt-4o
ANSWER_PROBABILITY=0.025

# Mutable state dir inside the container (matches the ntk_data volume mount).
DATA_DIR=/data
```

- [ ] **Step 2: Create `infra/deploy.sh`**

```bash
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
```

- [ ] **Step 3: Make it executable and sanity-check syntax**

```bash
chmod +x infra/deploy.sh
bash -n infra/deploy.sh && echo "syntax ok"
```
Expected: prints `syntax ok`

- [ ] **Step 4: Commit**

```bash
git add infra/.env.production.example infra/deploy.sh
git commit -m "feat(infra): env template and remote deploy script"
```

---

## Part C — CI/CD workflows

### Task 14: CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  pull_request:
    branches: [master]
  push:
    branches: [master]
  workflow_dispatch:

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

permissions:
  contents: read

jobs:
  quality:
    name: Lint, Type, Test, Build
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v6

      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          version: "0.5.x"

      - name: Sync dependencies
        run: uv sync --frozen

      - name: Ruff format check
        run: uv run ruff format --check .

      - name: Ruff lint
        run: uv run ruff check .

      - name: Type check
        run: uv run ty check

      - name: Tests
        run: uv run pytest

      - name: Validate compose file
        run: |
          set -euo pipefail
          cat > /tmp/ntk.env <<'EOF'
          NTK_BOT_IMAGE=ghcr.io/vsem-azamat/ntk-bot:ci
          BOT_TOKEN=ci-token
          SUPER_ADMINS=
          DELTA_TIME=20
          OPENROUTER_API_KEY=
          OPENROUTER_MODEL=openai/gpt-4o
          ANSWER_PROBABILITY=0.025
          DATA_DIR=/data
          EOF
          docker compose --env-file /tmp/ntk.env -f infra/docker-compose.prod.yml config >/dev/null

      - name: Validate Docker build
        run: docker build -f infra/Dockerfile -t ntk-bot:ci .
```

- [ ] **Step 2: Lint the YAML locally**

Run: `uv run python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml')); print('yaml ok')"`
Expected: prints `yaml ok` (uses PyYAML if present; otherwise skip — CI will validate)

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: lint, type-check, test, compose + docker build gates"
```

---

### Task 15: Deploy workflow

**Files:**
- Create: `.github/workflows/deploy.yml`

- [ ] **Step 1: Create `.github/workflows/deploy.yml`**

```yaml
name: Deploy

on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]
    branches: [master]
  workflow_dispatch:

concurrency:
  group: deploy
  cancel-in-progress: false

permissions:
  contents: read
  packages: write

jobs:
  deploy:
    name: Build image and deploy to azamat VPS
    runs-on: ubuntu-latest
    if: >-
      github.event_name == 'workflow_dispatch' ||
      github.event.workflow_run.conclusion == 'success'
    environment: production
    env:
      REGISTRY: ghcr.io
      IMAGE_NAME: ${{ github.repository }}
      SSH_HOST: ${{ secrets.DEMO_SSH_HOST }}
      SSH_USER: ${{ secrets.DEMO_SSH_USER }}
      SSH_PORT: ${{ secrets.DEMO_SSH_PORT || '22' }}
      DEPLOY_DIR: ${{ secrets.NTK_DEPLOY_DIR || '/home/azamat/deploy/ntk-bot' }}
      BOT_TOKEN: ${{ secrets.BOT_TOKEN }}
      SUPER_ADMINS: ${{ secrets.SUPER_ADMINS }}
      DELTA_TIME: ${{ secrets.DELTA_TIME || '20' }}
      OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
      OPENROUTER_MODEL: ${{ secrets.OPENROUTER_MODEL || 'openai/gpt-4o' }}
      ANSWER_PROBABILITY: ${{ secrets.ANSWER_PROBABILITY || '0.025' }}
    steps:
      - name: Checkout workflow run SHA
        if: github.event_name == 'workflow_run'
        uses: actions/checkout@v6
        with:
          ref: ${{ github.event.workflow_run.head_sha }}

      - name: Checkout dispatch SHA
        if: github.event_name == 'workflow_dispatch'
        uses: actions/checkout@v6

      - name: Validate deployment secrets
        env:
          SSH_KEY_PRESENT: ${{ secrets.DEMO_SSH_KEY != '' }}
        run: |
          set -euo pipefail
          missing=""
          [ -z "$SSH_HOST" ] && missing="$missing DEMO_SSH_HOST"
          [ -z "$SSH_USER" ] && missing="$missing DEMO_SSH_USER"
          [ "$SSH_KEY_PRESENT" != "true" ] && missing="$missing DEMO_SSH_KEY"
          [ -z "$BOT_TOKEN" ] && missing="$missing BOT_TOKEN"
          if [ -n "$missing" ]; then
            echo "::error::Missing required deployment secrets:$missing"
            exit 1
          fi

      - name: Validate deploy target
        run: |
          set -euo pipefail
          if [ "$DEPLOY_DIR" != "/home/azamat/deploy/ntk-bot" ]; then
            echo "::error::DEPLOY_DIR must be /home/azamat/deploy/ntk-bot"
            exit 1
          fi

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Docker metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=sha,prefix=
            type=raw,value=latest

      - name: Build and push image
        uses: docker/build-push-action@v6
        with:
          context: .
          file: infra/Dockerfile
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Install rsync
        run: |
          set -euo pipefail
          sudo apt-get update
          sudo apt-get install -y --no-install-recommends rsync

      - name: Configure SSH
        env:
          SSH_KEY: ${{ secrets.DEMO_SSH_KEY }}
        run: |
          set -euo pipefail
          mkdir -p ~/.ssh
          chmod 700 ~/.ssh
          printf '%s\n' "$SSH_KEY" > ~/.ssh/deploy_key
          chmod 600 ~/.ssh/deploy_key
          ssh-keyscan -p "$SSH_PORT" "$SSH_HOST" >> ~/.ssh/known_hosts

      - name: Prepare remote directory
        run: |
          set -euo pipefail
          ssh -i ~/.ssh/deploy_key -p "$SSH_PORT" \
            "$SSH_USER@$SSH_HOST" "mkdir -p '$DEPLOY_DIR/infra'"

      - name: Sync deployment files
        run: |
          set -euo pipefail
          rsync -az --delete \
            -e "ssh -i ~/.ssh/deploy_key -p $SSH_PORT" \
            infra/ "$SSH_USER@$SSH_HOST:$DEPLOY_DIR/infra/"

      - name: Render remote .env from secrets
        run: |
          set -euo pipefail
          tmp_env="$(mktemp)"
          chmod 600 "$tmp_env"
          {
            echo "NTK_BOT_IMAGE=${REGISTRY}/${IMAGE_NAME}:latest"
            echo "BOT_TOKEN=${BOT_TOKEN}"
            echo "SUPER_ADMINS=${SUPER_ADMINS}"
            echo "DELTA_TIME=${DELTA_TIME}"
            echo "OPENROUTER_API_KEY=${OPENROUTER_API_KEY}"
            echo "OPENROUTER_MODEL=${OPENROUTER_MODEL}"
            echo "ANSWER_PROBABILITY=${ANSWER_PROBABILITY}"
            echo "DATA_DIR=/data"
          } > "$tmp_env"
          rsync -az --chmod=F600 \
            -e "ssh -i ~/.ssh/deploy_key -p $SSH_PORT" \
            "$tmp_env" "$SSH_USER@$SSH_HOST:$DEPLOY_DIR/.env"
          rm -f "$tmp_env"

      - name: Deploy compose stack
        run: |
          set -euo pipefail
          ssh -i ~/.ssh/deploy_key -p "$SSH_PORT" \
            "$SSH_USER@$SSH_HOST" \
            "cd '$DEPLOY_DIR' && chmod +x infra/deploy.sh && ./infra/deploy.sh"
```

- [ ] **Step 2: Validate YAML**

Run: `uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/deploy.yml')); print('yaml ok')"`
Expected: prints `yaml ok` (or skip if PyYAML absent)

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/deploy.yml
git commit -m "ci: build/push image to GHCR and deploy to azamat VPS"
```

---

### Task 16: Deployment runbook + README

**Files:**
- Create: `docs/deployment/prod.md`
- Modify: `README.md`

- [ ] **Step 1: Create `docs/deployment/prod.md`**

```markdown
# ntk-bot Production Deployment

The bot is a single Telegram long-polling worker deployed to the `azamat` VPS
via GitHub Actions. There is no inbound port or public URL.

## Runtime

- Host: `azamat` (`46.225.117.31`), SSH user `azamat`
- Deploy directory: `/home/azamat/deploy/ntk-bot`
- Compose project: `ntk-bot`
- Compose file: `infra/docker-compose.prod.yml`
- Remote env: `/home/azamat/deploy/ntk-bot/.env`
- Image: `ghcr.io/vsem-azamat/ntk-bot:latest`
- Persistent volume: `ntk-bot_ntk_data` mounted at `/data`
  (`ntk.sqlite`, `model_*.pkl`, `.instructions`, `heartbeat`)

## Workflows

- `.github/workflows/ci.yml` — ruff format/lint, ty, pytest, compose + docker build.
- `.github/workflows/deploy.yml` — after CI success on `master` (or manual
  dispatch): build & push the image to GHCR, rsync `infra/`, render `.env`
  from secrets, run `infra/deploy.sh`.

## Required GitHub secrets

Reused from the shared azamat VPS access:

```
DEMO_SSH_HOST=46.225.117.31
DEMO_SSH_USER=azamat
DEMO_SSH_KEY=<private key for the azamat user>
DEMO_SSH_PORT=22            # optional
```

Bot-specific:

```
BOT_TOKEN=<telegram token, same as legacy>
SUPER_ADMINS=<comma-separated telegram ids>
OPENROUTER_API_KEY=<optional; empty disables GPT replies>
OPENROUTER_MODEL=<optional, default openai/gpt-4o>
DELTA_TIME=20               # optional
ANSWER_PROBABILITY=0.025    # optional
NTK_DEPLOY_DIR=/home/azamat/deploy/ntk-bot  # optional, guarded
```

The azamat host must be logged in to GHCR if the package is private:
`docker login ghcr.io`.

## First-time cutover from the legacy poryadok VPS

The legacy bot polls the same token, so it MUST be stopped before the new
container starts (one `getUpdates` consumer per token, else HTTP 409).

1. Provision the GitHub secrets above and confirm CI is green on `master`.
2. Pull the latest legacy data locally and stage it for azamat:
   ```sh
   ssh poryadok "tmux send-keys -t ntk C-c"      # stop the legacy python -m bot
   ssh poryadok "pkill -f 'python3 -m bot' || true"
   rsync -az poryadok:projects/NTK-bot/ntk_data.txt   /tmp/ntk_data.txt
   rsync -az poryadok:projects/NTK-bot/.instructions  /tmp/.instructions
   rsync -az /tmp/ntk_data.txt  azamat:/home/azamat/deploy/ntk-bot/legacy.txt
   rsync -az /tmp/.instructions azamat:/home/azamat/deploy/ntk-bot/.instructions
   ```
3. Trigger the deploy (merge to `master`, or run the Deploy workflow via
   `workflow_dispatch`). The container starts polling immediately; the DB is
   empty until the next step, but `/ntk` (live scrape) already works.
4. Import the legacy series and seed `.instructions` into the volume:
   ```sh
   ssh azamat '
     cd /home/azamat/deploy/ntk-bot
     cid=$(docker compose -f infra/docker-compose.prod.yml --env-file .env ps -q bot)
     docker cp legacy.txt        "$cid":/data/legacy.txt
     docker cp .instructions     "$cid":/data/.instructions
     docker exec "$cid" python scripts/import_legacy_ntk_data.py /data/legacy.txt
   '
   ```
   Expected output: `imported=~105000 skipped=<small> total_rows_in_db=~105000`.
5. Retrain the models on-host (in Telegram, as a super admin: `/learn`), or:
   ```sh
   ssh azamat 'cd /home/azamat/deploy/ntk-bot && \
     docker exec "$(docker compose -f infra/docker-compose.prod.yml --env-file .env ps -q bot)" \
     python -c "import asyncio; from apps.predictModels import predictModels; asyncio.run(predictModels.learn_models())"'
   ```
6. Verify in the chat: `/ntk`, `/graph` (shows history + prediction lines),
   `/weather`. Confirm new rows append: re-run the importer's count after a
   collection window, or `docker exec "$cid" python -c "from bot import db; print(len(db.iter_rows()))"`.
7. Decommission legacy: archive `~/projects/NTK-bot/ntk_data.txt` on poryadok,
   kill the `ntk` tmux session (`tmux kill-session -t ntk`).

## Note on the GPT key

The legacy `.env` used `OPENAI_API_KEY` (direct OpenAI). The modern code uses
OpenRouter. The old key will NOT work — set a real `OPENROUTER_API_KEY` secret
to enable GPT replies, or leave it empty to ship with GPT disabled.
```

- [ ] **Step 2: Update `README.md` deployment section**

Replace the `## Installation and start` "Start" note's deployment context by
appending a new section after the existing `## Commands:` section:

```markdown

## Deployment

Production runs on the `azamat` VPS as a Docker Compose stack, delivered by
GitHub Actions (build → GHCR → SSH deploy). Occupancy data lives in SQLite on a
persistent volume (`/data/ntk.sqlite`). See
[`docs/deployment/prod.md`](docs/deployment/prod.md) for the runbook and the
one-time cutover from the legacy manual deployment.

Local dev still uses `uv run ntk-bot` with a `.env`; data is written to
`./ntk.sqlite` (overridable with `DATA_DIR`).
```

- [ ] **Step 3: Commit**

```bash
git add docs/deployment/prod.md README.md
git commit -m "docs: production deployment runbook and README deploy section"
```

---

### Task 17: Push branch and open PR

- [ ] **Step 1: Final full gate**

```bash
uv run ruff format --check . && uv run ruff check . && uv run ty check && uv run pytest
```
Expected: all pass.

- [ ] **Step 2: Push and open PR**

```bash
git push -u origin feature/cicd-deployment
gh pr create --base master --title "Professional CI/CD deployment + SQLite persistence" \
  --body "Implements docs/superpowers/specs/2026-06-09-ntk-bot-cicd-deployment-design.md"
```
Expected: PR created; CI runs and goes green.

- [ ] **Step 3: STOP — human gate before cutover**

Do NOT merge automatically. Merging to `master` triggers the deploy workflow,
which starts a second poller on the production token. Follow the cutover
sequence in `docs/deployment/prod.md` (stop legacy first) when ready to go live.

---

## Operational pre-reqs (outside the repo — do once before going live)

These are not code tasks but are required for the deploy to succeed. Track them
separately:

- [ ] Add GitHub repo/environment secrets listed in `docs/deployment/prod.md`.
- [ ] Create `production` GitHub Environment (matches `environment: production`).
- [ ] Ensure the `azamat` user can `docker login ghcr.io` (PAT with `read:packages`)
      if the GHCR package stays private; or make the package public.
- [ ] Confirm `DEMO_SSH_KEY` authorises the `azamat` user on `46.225.117.31`.

---

## Self-Review notes

- **Spec coverage:** SQLite layer (Tasks 1–9), DATA_DIR/models (Tasks 1,4),
  migration script (Task 8), Dockerfile (11), compose+healthcheck (12),
  env+deploy.sh (13), CI gates (14), deploy workflow (15), runbook+credential
  mapping+cutover (16). All spec sections mapped.
- **Tests stay green:** the string-row format, `parse_row_datetime`,
  `remove_zero_values`, and `model_filename` (via `Path(".")` normalisation)
  are preserved, so the three existing tests in `tests/apps/` are unaffected.
- **Type consistency:** `db.insert_occupancy(ts, people)`, `db.iter_rows()`,
  `db.export_text()`, and `scripts.import_legacy_ntk_data.import_file(path)`
  are referenced identically wherever they appear.
```
