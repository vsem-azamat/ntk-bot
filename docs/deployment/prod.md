# ntk-bot Production Deployment

The bot is a single Telegram long-polling worker deployed to the `azamat` VPS
via GitHub Actions. There is no inbound port or public URL.

## Runtime

- Host: `azamat` (`REDACTED`), SSH user `azamat`
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
DEMO_SSH_HOST=REDACTED
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
   ssh poryadok "pkill -f 'python3 -m bot' || true"   # stop the legacy bot
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
   `/weather`. Confirm new rows append after a collection window.
7. Decommission legacy: archive `~/projects/NTK-bot/ntk_data.txt` on poryadok,
   kill the `ntk` tmux session (`tmux kill-session -t ntk`).

## Note on the GPT key

The legacy `.env` used `OPENAI_API_KEY` (direct OpenAI). The modern code uses
OpenRouter. The old key will NOT work — set a real `OPENROUTER_API_KEY` secret
to enable GPT replies, or leave it empty to ship with GPT disabled.
