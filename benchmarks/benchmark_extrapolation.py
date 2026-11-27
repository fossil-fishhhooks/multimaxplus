"""Time exact/extrapolated Fehlberg on one host, launching MPI jobs itself.

Run directly with Python, NOT mpirun. Each timed batch is one continuous
steps(n), including the five-step predictor startup. A separate untimed probe
checks activation. Profiling is OFF during timings; CUDA completion is awaited.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from benchmark_suite import Driver, label, shape_arg, write_json
from demag_extrapolation import advance, make_case


MODES = {"normal": (1, False), "exact2": (2, False), "unchecked2": (2, True), "extrapolated2": (2, True)}


def worker(args):
    import numpy as np

    root = Path(__file__).resolve().parents[1]
    build = (root / args.build_dir).resolve()
    sys.path[:0] = [str(root), str(build / "src/bindings")]
    from mumaxplus import _cpp

    if not Path(_cpp.__file__).resolve().is_relative_to(build):
        raise RuntimeError(f"Wrong backend: {_cpp.__file__}")
    ranks, enabled = MODES[args.worker]
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
    if int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1")) != ranks:
        raise RuntimeError("Wrong MPI rank count")
    os.environ["MUMAXPLUS_EXTRAPOLATE_DEMAG"] = str(int(enabled))
    # Explicit settings prevent inherited shell flags changing the comparison.
    os.environ["MUMAXPLUS_PRUNE_DEMAG"] = "1"
    os.environ["MUMAXPLUS_LOCAL_RK"] = "1"
    os.environ["MUMAXPLUS_DEMAG_CHECK"] = "0" if args.worker == "unchecked2" else "1"
    os.environ["MUMAXPLUS_DEMAG_ATOL"] = str(args.atol)
    os.environ["MUMAXPLUS_DEMAG_RTOL"] = str(args.rtol)
    os.environ["MUMAXPLUS_PROFILE_STEPS"] = "1"
    world, magnet, solver = make_case(args.grid, args.case, args.cellsize_nm)
    solver.timestep = args.dt_ps * 1e-12
    driver = Driver()
    stats = advance(solver, 6, details=True)
    exact, predicted = stats["exact"], stats["predicted"]
    expected = 36 + 6 * stats["retries"] + 2 * stats["checks"]
    if (enabled and (predicted == 0 or exact + predicted != expected or
                     (args.worker == "extrapolated2" and stats["checks"] == 0))) or (not enabled and predicted):
        raise RuntimeError(f"Activation mismatch: exact/predicted={exact}/{predicted}")
    os.environ["MUMAXPLUS_PROFILE_STEPS"] = "0"
    solver.steps(args.warmup)
    driver.sync()
    intervals = []
    for _ in range(args.batches):
        start = time.perf_counter_ns()
        solver.steps(args.steps)
        driver.sync()
        intervals.append((start, time.perf_counter_ns()))
    mean = np.asarray(magnet.magnetization.average(), dtype=float)
    if not np.isfinite(mean).all():
        raise RuntimeError("Nonfinite final magnetization")
    write_json(args.output / f"rank-{rank}.json", dict(
        rank=rank, mode=args.worker, backend=str(_cpp.__file__),
        intervals=intervals, mean_m=mean.tolist(), final_time_s=solver.time,
        probe_exact=exact, probe_predicted=predicted, probe_checks=stats["checks"], probe_retries=stats["retries"],
    ))


def batch_times(records, steps):
    """Use the full rank span, with the common single-host monotonic clock."""
    count = len(records[0]["intervals"])
    if any(len(r["intervals"]) != count for r in records):
        raise RuntimeError("Rank batch counts disagree")
    return [
        (max(r["intervals"][i][1] for r in records)
         - min(r["intervals"][i][0] for r in records)) / steps / 1e6
        for i in range(count)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build-a100"))
    parser.add_argument("--sizes", type=shape_arg, nargs="+", default=[(512, 512, 32), (1024, 1024, 32)])
    parser.add_argument("--dt-ps", type=float, default=0.1)
    parser.add_argument("--case", choices=("relaxation", "reversal", "domain_wall", "spin_wave", "material", "shaped"), default="relaxation")
    parser.add_argument("--cellsize-nm", type=float, default=2.0)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument("--grid", type=shape_arg, help=argparse.SUPPRESS)
    parser.add_argument("--multigpu", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.steps < 6 or args.warmup < 0 or args.batches < 1:
        parser.error("Need steps >= 6, warmup >= 0, batches >= 1")
    if not (0 < args.dt_ps < float("inf")):
        parser.error("dt must be finite and positive")
    if not (0 <= args.atol < float("inf") and 0 <= args.rtol < float("inf") and 0 < args.cellsize_nm < float("inf")):
        parser.error("Invalid tolerances or cellsize")
    if args.worker:
        worker(args)
        return
    if "OMPI_COMM_WORLD_RANK" in os.environ:
        parser.error("Run directly with Python, not mpirun")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "settings.json", dict(
        dt_ps=args.dt_ps, atol=args.atol, rtol=args.rtol, case=args.case, cellsize_nm=args.cellsize_nm, steps=args.steps, warmup=args.warmup, batches=args.batches,
        sizes=args.sizes, build_dir=str(args.build_dir),
        timing="single-host rank span; includes predictor startup in every batch",
    ))
    rows = []
    for grid in args.sizes:
        medians = {}
        for mode, (ranks, _) in MODES.items():
            job = args.output / label(grid) / mode
            job.mkdir(parents=True)
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--worker", mode, "--grid", label(grid),
                       "--build-dir", str(args.build_dir), "--output", str(job),
                       "--dt-ps", str(args.dt_ps), "--steps", str(args.steps),
                       "--warmup", str(args.warmup), "--batches", str(args.batches),
                       "--case", args.case, "--cellsize-nm", str(args.cellsize_nm),
                       "--atol", str(args.atol), "--rtol", str(args.rtol)]
            if ranks > 1:
                command.append("--multigpu")
                command = [args.mpirun, "--allow-run-as-root", "--bind-to", "none", "-n", str(ranks)] + command
            env = os.environ.copy()
            env.setdefault("NVSHMEM_SYMMETRIC_SIZE", "32G")
            print(f"{label(grid)} {mode} ...", flush=True)
            with (job / "run.log").open("w") as log:
                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Worker failed; see {job / 'run.log'}")
            records = [json.loads((job / f"rank-{r}.json").read_text()) for r in range(ranks)]
            intervals = batch_times(records, args.steps)
            median = medians[mode] = statistics.median(intervals)
            for batch, ms in enumerate(intervals, 1):
                rows.append(dict(grid=label(grid), mode=mode, batch=batch,
                                 ms_per_step=ms, median_ms=median))
            with (args.output / "timings.csv").open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"  median {median:.4f} ms/step; batches=" +
                  ", ".join(f"{v:.4f}" for v in intervals), flush=True)
        print(f"{label(grid)}: extrapolation benefit on 2 GPUs "
              f"{medians['exact2'] / medians['extrapolated2']:.2f}x; "
              f"versus exact 1 GPU {medians['normal'] / medians['extrapolated2']:.2f}x", flush=True)
    print(f"Results: {args.output / 'timings.csv'}")


if __name__ == "__main__":
    main()
