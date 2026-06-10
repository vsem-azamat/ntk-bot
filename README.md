# NTK bot
Telegram bot in @chat_ntk chat for students who regularly visit the National Technical Library. The bot regularly collects statistics on library visits. Based on this data, machine learning models predict the expected occupancy of the library — together with an uncertainty range, not just a single number.


<p align="center">
  <img src="example_images/daily_graph_with_predictions.jpg" alt="NTK occupancy with prediction" width="80%">
</p>


## Current and planned functions:
- [x] Shows the current number of people in the NTK
- [x] Regular storage of data from the library website on the number of people
- [x] Draws a diagram of people's visits in the NTK
- [x] Predicting the number of people in the library (median + p10–p90 range) with ML models
- [x] Uses weather as an input signal for the prediction
- [ ] Anti-bot filter
- [ ] Function for temporary self mute/ban from the chat so that students are not distracted from their studies

## Data sources:
- [NTK website](https://www.techlib.cz/)
- [Open-Meteo](https://open-meteo.com/)


## How prediction works:
Every occupancy sample is stored permanently. For each timestamp the model builds a feature vector and predicts the number of people:

| feature group | features |
|:---|:---|
| time (cyclical) | minute-of-day (sin/cos), day-of-year (sin/cos) |
| calendar | weekday, month, weekend flag |
| weather | temperature, precipitation, cloud cover, wind (joined by hour) |

Instead of a single number, three **quantile** models are trained, so the bot shows a range rather than just a point estimate:
- **p50** — the median ("usually about this many people")
- **p10 / p90** — the central 80% range ("normally somewhere between these")

The models are [CatBoost](https://catboost.ai/) gradient-boosted regressors (`loss=Quantile`). They are validated on a **time-based holdout** (the most recent weeks, never shuffled — so past predictions can't leak future data) and benchmarked against a simple **climatology baseline** (historical median per weekday × time-of-day). CatBoost ships only when it beats the baseline; otherwise — and on a cold start — the bot falls back to the baseline.

Historical weather is backfilled once from the Open-Meteo archive and kept current daily; the forecast for the target day feeds the prediction.

## Installation and start

The project is managed with [uv](https://docs.astral.sh/uv/).

### Necessary:
Install dependencies (creates a virtual environment from `uv.lock`)
```sh
> uv sync
```
Create a `.env` file and add the **bot token**
```env
BOT_TOKEN=<TOKEN>
```

### Start:
From the root directory of the project
```sh
> uv run ntk-bot
```
(equivalently `uv run python -m bot`)

### Optional:
Additional adjustable values in `.env`
```env
DELTA_TIME=<int>
SUPER_ADMINS=<int,int,int,...>
OPENROUTER_API_KEY=<KEY>
OPENROUTER_MODEL=<slug>
ANSWER_PROBABILITY=<float>
```
* `DELTA_TIME` - The time interval with which the bot collects visit data from the site. The default value is `20`
* `SUPER_ADMINS` - List of super admins for admin commands
* `OPENROUTER_API_KEY` - [OpenRouter](https://openrouter.ai/) key used for the random GPT replies
* `OPENROUTER_MODEL` - Model slug passed to OpenRouter. Default `openai/gpt-4o`
* `ANSWER_PROBABILITY` - Probability of a random GPT reply. Default `0.025`

### Development:
```sh
> uv run ruff format .      # format
> uv run ruff check .       # lint
> uv run ty check           # type-check
> uv run pytest             # tests
```


## Commands:
Prefixes: `!/`
- `/ntk` - Show the current number of people in the library
- `/help` - Show help
- `/graph` - Draw and send the occupancy graph: real data, the predicted median and the p10–p90 range
- `/learn` - Re-train the CatBoost prediction models on the stored data

## Deployment

Production runs on the `azamat` VPS as a Docker Compose stack, delivered by
GitHub Actions (build → GHCR → SSH deploy). Occupancy data lives in SQLite on a
persistent volume (`/data/ntk.sqlite`). See
[`docs/deployment/prod.md`](docs/deployment/prod.md) for the runbook and the
one-time cutover from the legacy manual deployment.

Local dev still uses `uv run ntk-bot` with a `.env`; data is written to
`./ntk.sqlite` (overridable with `DATA_DIR`).