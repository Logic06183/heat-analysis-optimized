"""Tests for mcd_pipeline.utils.reproducibility."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mcd_pipeline.utils.reproducibility import (
    set_global_seeds,
    capture_environment,
    compute_data_checksum,
    save_provenance,
)


class TestSetGlobalSeeds:

    def test_numpy_deterministic(self):
        set_global_seeds(42)
        a = np.random.random(10)
        set_global_seeds(42)
        b = np.random.random(10)
        np.testing.assert_array_equal(a, b)

    def test_pythonhashseed_set(self):
        set_global_seeds(123)
        assert os.environ["PYTHONHASHSEED"] == "123"

    def test_different_seeds_different_output(self):
        set_global_seeds(42)
        a = np.random.random(10)
        set_global_seeds(99)
        b = np.random.random(10)
        assert not np.array_equal(a, b)


class TestCaptureEnvironment:

    def test_returns_dict(self):
        env = capture_environment()
        assert isinstance(env, dict)

    def test_has_required_keys(self):
        env = capture_environment()
        assert "python_version" in env
        assert "platform" in env
        assert "numpy_version" in env
        assert "packages" in env
        assert "timestamp" in env

    def test_packages_is_list(self):
        env = capture_environment()
        assert isinstance(env["packages"], list)

    def test_numpy_version_realistic(self):
        env = capture_environment()
        assert env["numpy_version"].startswith(("1.", "2."))


class TestComputeDataChecksum:

    def test_known_content(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("a,b,c\n1,2,3\n")
            f.flush()
            path = Path(f.name)

        try:
            checksum = compute_data_checksum(path)
            assert isinstance(checksum, str)
            assert len(checksum) == 64  # SHA-256 hex digest
        finally:
            path.unlink()

    def test_deterministic(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("hello world\n")
            f.flush()
            path = Path(f.name)

        try:
            c1 = compute_data_checksum(path)
            c2 = compute_data_checksum(path)
            assert c1 == c2
        finally:
            path.unlink()

    def test_different_content_different_checksum(self):
        paths = []
        for content in ["data1\n", "data2\n"]:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                f.write(content)
                f.flush()
                paths.append(Path(f.name))

        try:
            c1 = compute_data_checksum(paths[0])
            c2 = compute_data_checksum(paths[1])
            assert c1 != c2
        finally:
            for p in paths:
                p.unlink()

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            compute_data_checksum(Path("/nonexistent/file.csv"))


class TestSaveProvenance:

    def test_creates_json_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a dummy data file
            data_file = tmpdir / "test_data.csv"
            data_file.write_text("a,b\n1,2\n")

            outpath = save_provenance(
                output_dir=tmpdir / "provenance",
                data_files={"test": data_file},
            )

            assert outpath.exists()
            assert outpath.suffix == ".json"

            with open(outpath) as f:
                prov = json.load(f)

            assert "environment" in prov
            assert "data_checksums" in prov
            assert "test" in prov["data_checksums"]
            assert "sha256" in prov["data_checksums"]["test"]

    def test_handles_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            outpath = save_provenance(
                output_dir=tmpdir,
                data_files={"missing": Path("/nonexistent.csv")},
            )

            with open(outpath) as f:
                prov = json.load(f)

            assert "error" in prov["data_checksums"]["missing"]
