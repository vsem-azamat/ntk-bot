from pathlib import Path

from decouple import config

# Directory that holds local runtime state (trained models, the outbox, the
# liveness heartbeat, the legacy files imported once). "." for local dev,
# "/data" (a Docker volume) in production.
DATA_DIR: str = config("DATA_DIR", cast=str, default=".")

BAD_WORDS_PATH = "bad_words.txt"
# Legacy LLM prompt file, imported once into the settings table if no prompt is stored.
LEGACY_INSTRUCTIONS_PATH = str(Path(DATA_DIR) / ".instructions")
# Pre-Postgres SQLite database, imported once on startup and then kept as a cold copy.
LEGACY_DB_PATH = str(Path(DATA_DIR) / "ntk.sqlite")
HEARTBEAT_PATH = str(Path(DATA_DIR) / "heartbeat")


def _load_lines(path: str) -> list[str]:
    """Return the stripped, non-empty lines of ``path`` (empty list if missing)."""
    try:
        with open(path, encoding="utf-8") as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        return []


class Config:
    # >>>>>>>>>> TELEGRAM <<<<<<<<<< #
    BOT_TOKEN: str = config("BOT_TOKEN", cast=str)

    ID_NTK_BIG_CHAT: int = -1001684546093
    ID_NTK_SMALL_CHAT: int = -1001384533622
    SUPER_ADMINS: list[int] = [
        int(id_admin)
        for id_admin in config("SUPER_ADMINS", cast=str, default="").split(",")
        if id_admin
    ]

    # >>>>>>>>>> ANON <<<<<<<<<< #
    ANON_ENABLED: bool = True
    REVEAL_ANON_PROBABILITY: float = 0.1

    # >>>>>>>>>> FILES <<<<<<<<<< #
    BAD_WORDS: list[str] = _load_lines(BAD_WORDS_PATH)
    DATA_DIR: str = DATA_DIR
    LEGACY_DB_PATH: str = LEGACY_DB_PATH
    LEGACY_INSTRUCTIONS_PATH: str = LEGACY_INSTRUCTIONS_PATH
    HEARTBEAT_PATH: str = HEARTBEAT_PATH

    # >>>>>>>>>> DATABASE <<<<<<<<<< #
    # libpq URL of the PostgreSQL database (sslmode/sslrootcert/hostaddr go in
    # the query string in production).
    DATABASE_URL: str = config("DATABASE_URL", cast=str, default="")

    # >>>>>>>>>> PARSERS <<<<<<<<<< #
    DELTA_TIME_FOR_RECIEVE_NTK: int = config("DELTA_TIME", cast=int, default=20)

    # >>>>>>>>>> MORNING DIGEST <<<<<<<<<< #
    # A single daily message (occupancy forecast) posted at 06:45, refreshed
    # hourly, and deleted at 16:00. Off by default so it never posts unattended.
    DIGEST_ENABLED: bool = config("DIGEST_ENABLED", cast=bool, default=False)
    DIGEST_CHAT_ID: int = config("DIGEST_CHAT_ID", cast=int, default=ID_NTK_BIG_CHAT)

    # >>>>>>>>>> ALERTS <<<<<<<<<< #
    # Chat (and optional forum topic) for collector outage / recovery alerts.
    # 0 disables delivery; alerts are then only logged.
    ALERT_CHAT_ID: int = config("ALERT_CHAT_ID", cast=int, default=0)
    ALERT_THREAD_ID: int = config("ALERT_THREAD_ID", cast=int, default=0)

    # >>>>>>>>>> OPENROUTER <<<<<<<<<< #
    OPENROUTER_API_KEY: str = config("OPENROUTER_API_KEY", cast=str, default="")
    OPENROUTER_MODEL: str = config("OPENROUTER_MODEL", cast=str, default="openai/gpt-6-luna-pro")
    # Default reply probability until one is chosen with /llm (stored in the DB).
    LLM_ANSWER_PROBABILITY: float = config("ANSWER_PROBABILITY", cast=float, default=0.025)


cnfg = Config()
