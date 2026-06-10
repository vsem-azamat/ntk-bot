import asyncio

from .bot import main


def run() -> None:
    """Synchronous entrypoint used by the ``ntk-bot`` console script."""
    asyncio.run(main())


__all__ = ["run"]
