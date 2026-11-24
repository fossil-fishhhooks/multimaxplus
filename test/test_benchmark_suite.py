import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location(
    "suite", Path(__file__).parent.parent / "benchmarks/benchmark_suite.py"
)
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


def records(shape=(4, 4, 4), mode="cufftmp2", kind="dynamics", phase="timing"):
    result = []
    for rank in range(suite.MODES[mode][0]):
        result.append(
            {
                "rank": rank,
                "ranks": suite.MODES[mode][0],
                "hostname": "same-host",
                "backend_sha256": "hash",
                "precision": "SINGLE",
                "iterations": 10,
                "shape": list(shape),
                "kind": kind,
                "mode": mode,
                "phase": phase,
                "setup_start_ns": rank * 10000000,
                "setup_end_ns": (100 + 20 * rank) * 1000000,
                "batches": [
                    {
                        "start_ns": (100 + 5 * rank) * 1000000,
                        "end_ns": (150 + 10 * rank) * 1000000,
                    },
                    {
                        "start_ns": (200 + 5 * rank) * 1000000,
                        "end_ns": (250 + 20 * rank) * 1000000,
                    },
                ],
            }
        )
    return result


class Tests(unittest.TestCase):

    def test_shapes_and_fast_path(self):
        self.assertEqual(suite.fft_shape((3, 3, 1)), (5, 5, 1))
        self.assertEqual(suite.fft_shape((8, 6, 4)), (16, 12, 8))
        self.assertFalse(suite.pruning_expected((8, 6, 1), "pruned2"))
        self.assertFalse(suite.pruning_expected((8, 6, 5), "pruned2"))
        self.assertTrue(suite.pruning_expected((8, 6, 4), "pruned2"))
        self.assertFalse(suite.pruning_expected((8, 6, 4), "cufftmp2"))
        with self.assertRaises(argparse.ArgumentTypeError):
            suite.shape_arg("3x0x5")

    def test_rank_timing_uses_job_span(self):
        summary = suite.summarize(records())
        self.assertEqual(summary["batch_ms"], [6, 7])
        self.assertEqual(summary["median_ms"], 6.5)
        self.assertAlmostEqual(summary["setup_s"], 0.12)
        self.assertAlmostEqual(summary["p10_ms"], 6.1)
        self.assertAlmostEqual(summary["p90_ms"], 6.9)

    def test_mismatched_or_missing_ranks_rejected(self):
        for key, value in [
            ("hostname", "another-host"),
            ("backend_sha256", "other"),
            ("precision", "DOUBLE"),
            ("iterations", 12),
        ]:
            r = records()
            r[1][key] = value
            with self.assertRaises(RuntimeError):
                suite.summarize(r)
        with self.assertRaises(RuntimeError):
            suite.summarize(records()[:1])

    def test_all_ranks_compared_and_failed_files_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, actual = (root / "reference", root / "candidate")
            reference.mkdir()
            actual.mkdir()
            field = np.ones((3, 2, 2, 2), dtype=np.float32)
            np.save(reference / "field-rank-0.npy", field)
            np.save(actual / "field-rank-0.npy", field)
            np.save(actual / "field-rank-1.npy", field)
            self.assertEqual(suite.compare_fields(reference, actual, 2), 0)
            changed = field.copy()
            changed[0, 0, 0, 0] += 0.01
            np.save(actual / "field-rank-1.npy", changed)
            with self.assertRaises(RuntimeError):
                suite.compare_fields(reference, actual, 2)
            self.assertTrue((actual / "field-rank-1.npy").exists())
            changed[0, 0, 0, 0] = np.nan
            np.save(actual / "field-rank-1.npy", changed)
            with self.assertRaises(RuntimeError):
                suite.compare_fields(reference, actual, 2)
            (actual / "keep.json").write_text("{}")
            suite.remove_fields(actual)
            self.assertTrue((actual / "keep.json").exists())
            self.assertTrue((reference / "field-rank-0.npy").exists())

    def test_checkpoints_drop_warmup_and_allow_zero_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory)
            (job / "rank-0.log").write_text(
                "header\n"
                + "".join(
                    (
                        f"step-profile rank=0 t=1e-13 attempts=1 demag={v} error=0 ms\n"
                        for v in [100, 200, 4, 6]
                    )
                )
            )
            r = records(mode="normal", phase="checkpoints")[0]
            r["iterations"] = 2
            r["batches"] = r["batches"][:1]
            rows = suite.checkpoint_rows(job, [r])
            self.assertEqual(
                next((x for x in rows if x["stage"] == "demag"))["median_ms"], 5
            )
            self.assertEqual(
                next((x for x in rows if x["stage"] == "error"))["median_ms"], 0
            )

    def test_timeout_terminates_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(subprocess.TimeoutExpired):
                suite.execute(
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    os.environ.copy(),
                    Path(directory) / "log",
                    0.05,
                )


class BaselineTests(unittest.TestCase):
    def test_original_uses_separate_build(self):
        a = argparse.Namespace(
            repo=Path("/fork"),
            build_dir=Path("build"),
            original_repo=Path("/original"),
            original_build_dir=Path("build-clean"),
        )
        self.assertEqual(
            suite.build_for(a, "normal"), (Path("/fork"), Path("/fork/build"))
        )
        self.assertEqual(
            suite.build_for(a, "original"),
            (Path("/original"), Path("/original/build-clean")),
        )


if __name__ == "__main__":
    unittest.main()
