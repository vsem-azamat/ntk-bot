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


def _load_lines(path: str) -> list[str]:
    """Return the stripped, non-empty lines of ``path`` (empty list if missing)."""
    try:
        with open(path, encoding="utf-8") as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        return []


def _load_text(path: str) -> str:
    """Return the contents of ``path`` (empty string if missing)."""
    try:
        with open(path, encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return ""


class Config:
    # >>>>>>>>>> TELEGRAM <<<<<<<<<< #
    BOT_TOKEN: str = config("BOT_TOKEN", cast=str)

    ID_NTK_BIG_CHAT: int = -1001684546093
    ID_NTK_SMALL_CHAT: int = -1001384533622
    ID_NTK_CHANNEL: int = -1001918057675
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
    DB_PATH: str = DB_PATH
    HEARTBEAT_PATH: str = HEARTBEAT_PATH

    # >>>>>>>>>> PARSERS <<<<<<<<<< #
    DELTA_TIME_FOR_RECIEVE_NTK: int = config("DELTA_TIME", cast=int, default=20)

    # >>>>>>>>>> OPENROUTER <<<<<<<<<< #
    OPENROUTER_API_KEY: str = config("OPENROUTER_API_KEY", cast=str, default="")
    OPENROUTER_MODEL: str = config("OPENROUTER_MODEL", cast=str, default="openai/gpt-4o")
    INSTRUCTIONS: str = _load_text(INSTRUCTIONS_PATH)
    GPT_ANSWER_PROBABILITY: float = config("ANSWER_PROBABILITY", cast=float, default=0.025)


cnfg = Config()
