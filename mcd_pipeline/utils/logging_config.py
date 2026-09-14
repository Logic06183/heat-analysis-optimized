"""
Logging Configuration — Standardised Pipeline Logging
======================================================

Provides consistent logging setup across all pipeline modules.
Logs to both console (INFO) and file (DEBUG) in mcd_outputs/logs/.
"""

import logging
from pathlib import Path

from mcd_pipeline import config


def setup_logging(
    name: str = "mcd_pipeline",
    log_dir: Path = config.OUTPUT_ROOT / "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure and return a logger with file and console handlers.

    Parameters
    ----------
    name : str
        Logger name (typically the module name).
    log_dir : Path
        Directory for log files.
    level : int
        Console logging level.

    Returns
    -------
    logging.Logger
    """
    raise NotImplementedError("Utils: setup_logging")
