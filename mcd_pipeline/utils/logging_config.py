"""
Logging Configuration — Standardised Pipeline Logging
======================================================

Provides consistent logging setup across all pipeline modules.
Logs to both console (INFO) and file (DEBUG) in mcd_outputs/logs/.
"""

import logging
from pathlib import Path

from mcd_pipeline import config


def configure_logging(
    name: str = "mcd_pipeline",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure root logging for standalone CLI entry-points.

    Writes to both stderr and ``mcd_outputs/logs/pipeline.log``.
    Safe to call multiple times — ``basicConfig`` with ``force=True``
    replaces any previously registered handlers.
    """
    log_dir = config.OUTPUT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "pipeline.log"),
        ],
    )
    return logging.getLogger(name)


def setup_logging(
    name: str = "mcd_pipeline",
    log_dir: Path = config.OUTPUT_ROOT / "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return a logger with file and console handlers.

    Thin wrapper around ``configure_logging`` kept for backward compatibility.
    """
    return configure_logging(name=name, level=level)
