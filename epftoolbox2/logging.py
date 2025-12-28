import logging
from rich.console import Console
from rich.logging import RichHandler

_console = Console()
_configured_loggers = set()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if name not in _configured_loggers:
        logger.addHandler(RichHandler(console=_console, rich_tracebacks=True))
        logger.setLevel(logging.INFO)
        _configured_loggers.add(name)

    return logger
