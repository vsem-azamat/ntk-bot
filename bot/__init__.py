import asyncio


def run() -> None:
    """Synchronous entrypoint used by the ``ntk-bot`` console script."""
    from .bot import main

    asyncio.run(main())


__all__ = ["run"]
