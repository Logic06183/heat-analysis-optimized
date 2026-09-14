"""
Reproducibility Manager — Seeds, Environment, Provenance
=========================================================

Ensures reproducible results across runs:
- Fixed random seeds for numpy, sklearn, xgboost
- Environment capture (pip freeze, platform info)
- Data checksums for input verification
- Session logging with timestamps

Adapted from archived reference: _archive/sep2025_scripts/reproducibility_manager.py
"""

import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mcd_pipeline import config

logger = logging.getLogger(__name__)


def set_global_seeds(seed: int = config.MASTER_SEED) -> None:
    """Set random seeds for numpy, random, and environment variables.

    Sets:
    - ``PYTHONHASHSEED`` environment variable (for deterministic hashing)
    - ``random.seed()`` (stdlib)
    - ``numpy.random.seed()`` (legacy API, still used by sklearn internals)
    - ``numpy.random.default_rng()`` is NOT set globally — create per-use

    Parameters
    ----------
    seed : int
        Master seed value (default: config.MASTER_SEED = 42).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    logger.info(f"Global seeds set to {seed}")


def capture_environment() -> dict:
    """Capture current Python environment for provenance tracking.

    Returns
    -------
    dict
        Keys: python_version, platform, machine, numpy_version,
        packages (pip freeze output), timestamp, seed.
    """
    # Get installed packages via pip freeze
    try:
        pip_output = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.DEVNULL,
            timeout=30,
        ).decode().strip()
        packages = pip_output.split("\n")
    except (subprocess.SubprocessError, OSError):
        packages = ["<pip freeze failed>"]

    env = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "n_packages": len(packages),
        "packages": packages,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "master_seed": config.MASTER_SEED,
    }

    logger.info(
        f"Environment: Python {sys.version_info.major}.{sys.version_info.minor}."
        f"{sys.version_info.micro}, NumPy {np.__version__}, "
        f"{env['n_packages']} packages"
    )

    return env


def compute_data_checksum(filepath: Path, algorithm: str = "sha256") -> str:
    """Compute checksum of a data file for integrity verification.

    Reads the file in 64 KB chunks to handle large CSVs without
    loading everything into memory.

    Parameters
    ----------
    filepath : Path
        Path to the data file.
    algorithm : str
        Hash algorithm (default: sha256).

    Returns
    -------
    str
        Hex digest string.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Cannot checksum: {filepath}")

    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)

    digest = h.hexdigest()
    logger.info(f"Checksum ({algorithm}): {filepath.name} -> {digest[:16]}...")
    return digest


def save_provenance(
    output_dir: Path,
    data_files: dict[str, Path],
    extra_metadata: dict | None = None,
) -> Path:
    """Save a provenance record for a pipeline run.

    Creates a JSON file with environment info, data checksums,
    and any extra metadata.

    Parameters
    ----------
    output_dir : Path
        Directory to save provenance file.
    data_files : dict
        Label -> filepath mapping for input data files to checksum.
    extra_metadata : dict, optional
        Additional metadata to include.

    Returns
    -------
    Path
        Path to the saved provenance JSON.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    provenance = {
        "environment": capture_environment(),
        "data_checksums": {},
        "metadata": extra_metadata or {},
    }

    for label, fpath in data_files.items():
        fpath = Path(fpath)
        if fpath.exists():
            provenance["data_checksums"][label] = {
                "path": str(fpath),
                "sha256": compute_data_checksum(fpath),
                "size_bytes": fpath.stat().st_size,
            }
        else:
            provenance["data_checksums"][label] = {
                "path": str(fpath),
                "error": "file not found",
            }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outpath = output_dir / f"provenance_{timestamp}.json"

    with open(outpath, "w") as f:
        json.dump(provenance, f, indent=2, default=str)

    logger.info(f"Provenance saved to {outpath}")
    return outpath
