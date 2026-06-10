# NTK bot

Telegram bot for [@chat_ntk](https://t.me/chat_ntk) — students of the National Technical Library in Prague. It tracks how busy the library is in real time and predicts the occupancy for the rest of the day, with an uncertainty range rather than a single number.

<p align="center">
  <img src="example_images/daily_graph_with_predictions.jpg" alt="NTK occupancy with prediction" width="80%">
</p>

## Features

- Shows the current number of people in the library (`/ntk`)
- Continuously records occupancy from the library website into SQLite
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

The models are [CatBoost](https://catboost.ai/) regressors (`loss=Quantile`), validated on a **time-based holdout** (the most recent weeks, never shuffled — so the past can't leak future data) and benchmarked against a **climatology baseline** (historical median per weekday × time-of-day). CatBoost ships only when it beats the baseline; otherwise — and on a cold start — the bot falls back to the baseline.

Historical weather is backfilled once from [Open-Meteo](https://open-meteo.com/) and refreshed daily; the forecast for the target day feeds the prediction.

## Run locally

The project is managed with [uv](https://docs.astral.sh/uv/).

```sh
uv sync                 # install deps from uv.lock
echo "BOT_TOKEN=<token>" > .env
uv run ntk-bot          # or: uv run python -m bot
```

Data is written to `./ntk.sqlite` (override the directory with `DATA_DIR`).

### Optional `.env` values

| key | default | meaning |
|:---|:---|:---|
| `DELTA_TIME` | `20` | minutes between occupancy samples |
| `SUPER_ADMINS` | — | comma-separated admin user IDs |
| `OPENROUTER_API_KEY` | — | [OpenRouter](https://openrouter.ai/) key for random GPT replies |
| `OPENROUTER_MODEL` | `openai/gpt-4o` | model slug for OpenRouter |
| `ANSWER_PROBABILITY` | `0.025` | chance of a random GPT reply |

### Development

```sh
uv run ruff format .    # format
uv run ruff check .     # lint
uv run ty check         # type-check
uv run pytest           # tests
```

## Commands

- `/ntk` — current number of people in the library
- `/graph` — occupancy chart: real data, predicted median and the p10–p90 range
- `/learn` — re-train the prediction models on stored data *(admin)*
- `/data` — export the occupancy series as text *(admin)*

## Deployment

Containerised with Docker Compose and deployed via GitHub Actions. Occupancy data is persisted in SQLite on a Docker volume.
