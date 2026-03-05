"""Logging setup."""
import logging


def setup_logging(level: str = "INFO", format: str | None = None) -> None:
    fmt = format or "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt)
