# Production deployment

The bot is a single Telegram long-polling worker. It has no inbound port and no
public URL — it only makes outbound calls (Telegram, the library site,
Open-Meteo). All host-specific values (server, user, paths, keys) live in GitHub
Actions **secrets**, never in the repo.

## Workflows

- `.github/workflows/ci.yml` — ruff format/lint, `ty`, pytest, compose config +
  docker build.
- `.github/workflows/deploy.yml` — after CI succeeds on `master` (or a manual
  `workflow_dispatch`): build & push the image to GHCR, sync `infra/`, render
  the remote `.env` from secrets, and bring the Compose stack up.

To deploy: merge to `master` (CI → deploy) or run the Deploy workflow manually.
Only one process may poll the bot token at a time (a second `getUpdates`
consumer returns HTTP 409), so never run a second instance against the same
token.

## Required secrets

Set these in the repository's Actions secrets (names only — values stay in
GitHub):

| secret | purpose |
|:---|:---|
| `DEMO_SSH_HOST` / `DEMO_SSH_USER` / `DEMO_SSH_KEY` / `DEMO_SSH_PORT` | SSH access to the deploy host |
| `NTK_DEPLOY_DIR` | deploy directory on the host |
| `BOT_TOKEN` | Telegram bot token |
| `SUPER_ADMINS` | comma-separated admin Telegram IDs |
| `OPENROUTER_API_KEY` | optional; empty disables GPT replies |
| `OPENROUTER_MODEL` | optional, default `openai/gpt-4o` |
| `DELTA_TIME` / `ANSWER_PROBABILITY` | optional tuning |

## Runtime

- Stack: `docker compose -f infra/docker-compose.prod.yml --env-file .env`.
- State lives on a Docker volume mounted at `/data`: `ntk.sqlite`, the trained
  model files, `.instructions`, and the liveness `heartbeat`.
- Container timezone is `Europe/Prague`, so collection windows and stored
  timestamps are local.

## Re-training

Models retrain on startup and can be retrained on demand by a super admin with
`/learn` in chat.
