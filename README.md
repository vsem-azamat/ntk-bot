# NTK bot

Telegram bot for [@chat_ntk](https://t.me/chat_ntk) — students of the National Technical Library in Prague. It tracks how busy the library is in real time and predicts the occupancy for the rest of the day, with an uncertainty range rather than a single number.

<p align="center">
  <img src="example_images/daily_graph_with_predictions.jpg" alt="NTK occupancy with prediction" width="80%">
</p>

## Features

- Shows the current number of people in the library (`/ntk`)
- Continuously records occupancy from the library website into PostgreSQL
- Plots the day: real data so far + predicted median + p10–p90 range (`/graph`)
- Predicts occupancy with ML models that also use weather as a signal

**Planned:** morning digest (auto-updating, self-deleting), anti-bot filter.

## How prediction works

Every occupancy sample is stored permanently. For each timestamp the model builds a feature vector and predicts the number of people:

| feature group | features |
|:---|:---|
| time (cyclical) | minute-of-day (sin/cos), day-of-year (sin/cos) |
| calendar | weekday, month, weekend flag |
| weather | temperature, precipitation, cloud cover, wind |

Instead of a single number, three **quantile** models are trained, so the bot shows a range rather than a point estimate:

- **p50** — the median ("usually about this many people")
- **p10 / p90** — the central 80% range ("normally somewhere between these")

The models are [CatBoost](https://catboost.ai/) regressors (`loss=Quantile`), validated on a **time-based holdout** (the most recent weeks, never shuffled — so the past can't leak future data) and benchmarked against a **recent-days baseline** (median curve of the last six days of the same type: Mon–Thu, Fri, Sat, Sun/holiday). The served model only changes when the challenger beats the incumbent by 5 %, so the forecast does not flip shape from day to day; on a cold start the bot uses the baseline.

Only samples from opening time (08:00 on weekdays, 10:00 on weekends and Czech holidays) until midnight are used: after closing the library counter keeps a residue of a few hundred "people" until it resets at opening.

Historical weather is backfilled once from [Open-Meteo](https://open-meteo.com/) and refreshed daily; the forecast for the target day feeds the prediction.

## Run locally

The project is managed with [uv](https://docs.astral.sh/uv/).

The bot needs PostgreSQL (any recent version; production runs 18).

```sh
uv sync                 # install deps from uv.lock
docker run -d --name ntk-pg -p 127.0.0.1:55432:5432 -e POSTGRES_PASSWORD=ntk postgres:18
printf 'BOT_TOKEN=<token>\nDATABASE_URL=postgresql://postgres:ntk@localhost:55432/postgres\n' > .env
uv run ntk-bot          # or: uv run python -m bot
```

Tables are created on the first connection. Trained models and the local outbox live in `DATA_DIR` (default `.`).

### Optional `.env` values

| key | default | meaning |
|:---|:---|:---|
| `DELTA_TIME` | `20` | minutes between occupancy samples |
| `SUPER_ADMINS` | — | comma-separated admin user IDs |
| `OPENROUTER_API_KEY` | — | [OpenRouter](https://openrouter.ai/) key for random LLM replies |
| `OPENROUTER_MODEL` | `openai/gpt-6-luna-pro` | model slug for OpenRouter |
| `ANSWER_PROBABILITY` | `0.025` | default chance of a random LLM reply until one is chosen with `/llm` |
| `ALERT_CHAT_ID` / `ALERT_THREAD_ID` | — | chat / forum topic for collector outage alerts |

### Development

```sh
uv run ruff format .    # format
uv run ruff check .     # lint
uv run ty check         # type-check
TEST_DATABASE_URL=postgresql://postgres:ntk@localhost:55432/postgres uv run pytest   # tests (empty DB per test)
```

## Commands

- `/ntk` — current number of people in the library
- `/graph` — occupancy chart: real data, predicted median and the p10–p90 range
- `/learn` — re-train the prediction models on stored data *(admin)*
- `/data` — export the occupancy series as text *(admin)*
- `/llm` — how often the bot answers chat messages with the LLM *(admin)*
- `/prompt` — show the LLM system prompt *(admin, private chat with the bot)*
- `/setprompt` — reply to a message or a `.txt` file with the new prompt *(admin, private chat with the bot)*

The LLM persona prompt and the reply probability live in the database (`settings` table), not in the repository: they change from Telegram without a deploy and are included in the database backups. The bot keeps them in memory, so chat messages never wait on the database. The previous prompt is kept as `llm_prompt_previous`.

## Deployment

Containerised with Docker Compose and deployed via GitHub Actions. Occupancy, weather and digest state live in PostgreSQL; if the database is unreachable, samples are buffered in a local SQLite outbox and flushed once it is back.
