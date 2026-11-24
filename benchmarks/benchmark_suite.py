#!/usr/bin/env python3
"""Compare original mumax+, this fork on one GPU, and distributed FFT modes.

Run directly with Python in the configured SDK environment. --original-repo and
--original-build-dir enable the optional unmodified mumax+ baseline. Without them,
only the fork is compared. --nsys adds warmed traces; checkpoints run separately.
Arrays are compared on every rank then deleted. Fixed Fehlberg dt=1e-13 s, T=0.
"""
import argparse
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time

DEFAULT_SIZES = (
    "256x256x1",
    "1024x1024x1",
    "2048x2048x1",
    "128x128x8",
    "256x256x32",
    "512x512x32",
    "768x768x32",
    "1024x1024x32",
)
MODES = {
    "original": (1, False, False),
    "normal": (1, False, False),
    "cufftmp1": (1, True, False),
    "cufftmp2": (2, True, False),
    "pruned2": (2, True, True),
}
ROOT = Path(__file__).resolve()


def shape_arg(text):
    """Parse a positive x/y/z grid size."""
    try:
        shape = tuple((int(n) for n in text.lower().replace(",", "x").split("x")))
        if len(shape) != 3 or min(shape) < 1:
            raise ValueError
        return shape
    except ValueError:
        raise argparse.ArgumentTypeError(
            "use three positive dimensions, e.g. 512x512x32"
        ) from None


def label(shape):
    """Format an x/y/z grid for filenames and reports."""
    return "x".join(map(str, shape))


def fft_shape(shape):
    """Return the current self-demag convolution dimensions."""
    return tuple((2 * n - 1 + (2 * n - 1 > 5) for n in shape))


def pruning_expected(shape, mode):
    """Mirror the current matching-grid fast-path guards; not runtime telemetry."""
    ranks, multi, prune = MODES[mode]
    padded = fft_shape(shape)
    if not (
        multi
        and prune
        and (ranks > 1)
        and (padded == tuple((2 * n for n in shape)))
        and (shape[2] % ranks == 0)
        and (padded[1] % ranks == 0)
    ):
        return False
    z, y, x = (shape[2] // ranks, padded[1] // ranks, padded[0] // 2 + 1)
    exchange = 3 * z * x * padded[1]
    return (
        3 * z * shape[1] * 2 * x <= 2**31 - 1
        and exchange <= 2**31 - 1
        and (2 * exchange // ranks <= 2**31 - 1)
        and (padded[2] * y * x <= 2**31 - 1)
        and (3 * z <= 65535)
        and (3 * z * x <= 32 * 65535)
        and (padded[1] <= 32 * 65535)
        and (x <= 32 * 65535)
    )


def write_json(path, value):
    """Atomically replace a JSON record."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    """Hash a file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, fraction):
    """Interpolate a percentile of a nonempty sample."""
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def distribution(values):
    """Summarize positive timing samples without assuming independence."""
    if not values or not all((math.isfinite(v) and v > 0 for v in values)):
        raise RuntimeError("Invalid timing samples")
    median, mean = (statistics.median(values), statistics.mean(values))
    return {
        "median_ms": median,
        "mean_ms": mean,
        "min_ms": min(values),
        "max_ms": max(values),
        "p10_ms": percentile(values, 0.1),
        "p90_ms": percentile(values, 0.9),
        "mad_ms": statistics.median((abs(v - median) for v in values)),
        "cv_percent": 100 * statistics.stdev(values) / mean if len(values) > 1 else 0,
        "samples": len(values),
    }


class Driver:
    """Small ctypes wrapper for context timing and profiler control."""

    def __init__(self):
        """Load the CUDA driver without creating a context."""
        self.lib = ctypes.CDLL("libcuda.so.1")

    def call(self, name, types=(), arguments=()):
        """Invoke a CUDA driver function and check its status."""
        function = getattr(self.lib, name)
        function.argtypes, function.restype = (list(types), ctypes.c_int)
        status = function(*arguments)
        if status:
            raise RuntimeError(f"{name} returned CUDA status {status}")

    def sync(self):
        """Finish pending work in the current CUDA context."""
        self.call("cuCtxSynchronize")

    def integer(self, name):
        """Read a scalar integer from the CUDA driver."""
        value = ctypes.c_int()
        self.call(name, (ctypes.POINTER(ctypes.c_int),), (ctypes.byref(value),))
        return value.value


def load_backend(args):
    """Import the requested build and reject another backend or precision."""
    sys.path[:0] = [str(args.repo), str(args.build_dir / "src/bindings")]
    from mumaxplus import FP_PRECISION, _cpp

    if FP_PRECISION != args.precision:
        raise RuntimeError(
            f"Loaded precision {FP_PRECISION}; requested {args.precision}"
        )
    backend = Path(_cpp.__file__).resolve()
    if not backend.is_relative_to(args.build_dir):
        raise RuntimeError(f"Loaded {backend}; expected backend under {args.build_dir}")
    return (_cpp, backend)


def field_metrics(field, normalized):
    """Check a field in bounded chunks and compute its component means."""
    import numpy as np

    values = field.reshape(3, -1)
    sums, norm_error = (np.zeros(3), 0.0)
    for start in range(0, values.shape[1], 1 << 19):
        block = values[:, start : start + (1 << 19)].astype(np.float64)
        if not np.isfinite(block).all():
            raise RuntimeError("Non-finite output")
        sums += block.sum(axis=1)
        if normalized:
            norm_error = max(
                norm_error,
                float(np.max(np.abs(np.sqrt((block * block).sum(axis=0)) - 1))),
            )
    if normalized and norm_error > 0.001:
        raise RuntimeError(f"Magnetization is not normalized: {norm_error}")
    return {"mean": (sums / values.shape[1]).tolist(), "max_norm_error": norm_error}


def worker(args):
    """Run one GPU job and save rank-local timings and output."""
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
    fd = os.open(
        args.job_dir / f"rank-{rank}.log", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 420
    )
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    import numpy as np

    _, backend = load_backend(args)
    from mumaxplus import FP_PRECISION, Ferromagnet, Grid, World

    ranks, multi, prune = MODES[args.mode]
    if (
        int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1")) != ranks
        or args.multigpu != multi
        or FP_PRECISION != args.precision
    ):
        raise RuntimeError("Worker rank count, multi-GPU flag, or precision mismatch")
    backend_hash = sha256(backend)
    if backend_hash != args.backend_sha256:
        raise RuntimeError("The backend changed during the benchmark")
    driver = Driver()
    nx, ny, nz = args.shape
    initial = None
    if args.kind == "demag":
        initial = np.empty(
            (3, nz, ny, nx),
            dtype=np.float32 if FP_PRECISION == "SINGLE" else np.float64,
        )
        x = np.arange(nx, dtype=initial.dtype)[None, None, :]
        y = np.arange(ny, dtype=initial.dtype)[None, :, None]
        z = np.arange(nz, dtype=initial.dtype)[:, None, None]
        initial[0] = 1 + 0.2 * np.sin(0.13 * x)
        initial[1] = 0.3 + 0.2 * np.cos(0.17 * y)
        initial[2] = 0.4 + 0.2 * np.sin(0.11 * (z + x))
    setup_start = time.perf_counter_ns()
    world = World((2e-09, 2e-09, 2e-09))
    magnet = Ferromagnet(world, Grid(args.shape))
    magnet.msat, magnet.aex, magnet.alpha, magnet.temperature = (
        800000.0,
        1.3e-11,
        0.1,
        0,
    )
    magnet.magnetization = initial if initial is not None else (1, 0.2, 0.1)
    del initial
    world.bias_magnetic_field = (-0.0246, 0.0043, 0.002)
    solver = world.timesolver
    solver.set_method("Fehlberg")
    solver.adaptive_timestep, solver.timestep = (False, args.dt)
    demag = magnet.demag_field
    demag._impl.exec()
    driver.sync()
    setup_end = time.perf_counter_ns()
    device = driver.integer("cuCtxGetDevice")
    if device != (rank if multi else 0):
        raise RuntimeError(f"Unexpected device {device} for rank {rank}")
    print(
        f"kind={args.kind} mode={args.mode} rank={rank} "
        f"device={device} backend={backend}",
        flush=True,
    )
    print(
        f"prune_requested={prune} "
        f"pruning_expected={pruning_expected(args.shape, args.mode)}",
        flush=True,
    )

    def operations(count):
        if args.kind == "dynamics":
            solver.steps(count)
        else:
            for _ in range(count):
                demag._impl.exec()

    operations(args.warmup)
    driver.sync()
    intervals = []
    if args.phase == "nsys":
        driver.call("cuProfilerStart")
    try:
        for _ in range(args.batches):
            start = time.perf_counter_ns()
            operations(args.iterations)
            driver.sync()
            intervals.append({"start_ns": start, "end_ns": time.perf_counter_ns()})
    finally:
        if args.phase == "nsys":
            driver.call("cuProfilerStop")
    field = np.asarray(
        magnet.magnetization.eval() if args.kind == "dynamics" else demag.eval()
    )
    driver.sync()
    if field.shape != (3, nz, ny, nx):
        raise RuntimeError(f"Unexpected field shape {field.shape}")
    metrics = field_metrics(field, args.kind == "dynamics")
    if args.kind == "dynamics":
        expected = (args.warmup + args.batches * args.iterations) * args.dt
        if abs(solver.time - expected) > max(1e-18, expected * 0.0001):
            raise RuntimeError(f"Final time {solver.time} differs from {expected}")
    np.save(args.job_dir / f"field-rank-{rank}.npy", field)
    record = {
        "rank": rank,
        "ranks": ranks,
        "device": device,
        "hostname": socket.gethostname(),
        "kind": args.kind,
        "mode": args.mode,
        "phase": args.phase,
        "shape": args.shape,
        "fft_shape": fft_shape(args.shape),
        "precision": FP_PRECISION,
        "backend": str(backend),
        "backend_sha256": backend_hash,
        "pruning_expected": pruning_expected(args.shape, args.mode),
        "setup_start_ns": setup_start,
        "setup_end_ns": setup_end,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "batches": intervals,
        "dt_s": args.dt,
        "final_time_s": solver.time,
        **metrics,
    }
    write_json(args.job_dir / f"rank-{rank}.json", record)
    for i, batch in enumerate(intervals):
        local_ms = (batch["end_ns"] - batch["start_ns"]) / args.iterations / 1000000.0
        print(f"batch={i + 1} local_ms={local_ms:.6f}")
    print(f"final_time_s={solver.time} mean={metrics['mean']}", flush=True)


def summarize(records):
    """Combine rank intervals using the single-host job span."""
    ranks = records[0]["ranks"]
    if sorted((r["rank"] for r in records)) != list(range(ranks)):
        raise RuntimeError("Missing or duplicated rank records")
    for key in (
        "hostname",
        "backend_sha256",
        "precision",
        "iterations",
        "shape",
        "kind",
        "mode",
        "phase",
    ):
        if any((r[key] != records[0][key] for r in records)):
            raise RuntimeError(f"Rank records disagree about {key}")
    count = len(records[0]["batches"])
    if any((len(r["batches"]) != count for r in records)):
        raise RuntimeError("Rank batch counts differ")
    batches = []
    for i in range(count):
        start = min((r["batches"][i]["start_ns"] for r in records))
        end = max((r["batches"][i]["end_ns"] for r in records))
        batches.append((end - start) / records[0]["iterations"] / 1000000.0)
    setup = (
        max((r["setup_end_ns"] for r in records))
        - min((r["setup_start_ns"] for r in records))
    ) / 1000000000.0
    return {**distribution(batches), "batch_ms": batches, "setup_s": setup}


def compare_fields(reference_job, candidate_job, ranks):
    """Compare every candidate rank with the normal field in bounded chunks."""
    import numpy as np

    reference = np.load(reference_job / "field-rank-0.npy", mmap_mode="r")
    rtol, atol = (5e-05, 5e-07) if reference.dtype.itemsize == 4 else (1e-10, 1e-12)
    maximum, scale = (0.0, 0.0)
    for rank in range(ranks):
        actual = np.load(candidate_job / f"field-rank-{rank}.npy", mmap_mode="r")
        if actual.shape != reference.shape or actual.dtype != reference.dtype:
            raise RuntimeError("Field shape or dtype differs from reference")
        a, b = (actual.reshape(-1), reference.reshape(-1))
        for start in range(0, a.size, 1 << 19):
            got = a[start : start + (1 << 19)].astype(np.float64)
            wanted = b[start : start + (1 << 19)].astype(np.float64)
            if not np.isfinite(got).all() or not np.isfinite(wanted).all():
                raise RuntimeError("Non-finite field in comparison")
            maximum = max(maximum, float(np.max(np.abs(got - wanted))))
            scale = max(scale, float(np.max(np.abs(wanted))))
    if maximum > atol + rtol * scale:
        raise RuntimeError(
            f"Field comparison failed: max error={maximum:.6e}, scale={scale:.6e}"
        )
    return maximum


def execute(command, env, log, timeout):
    """Run a subprocess group, terminating it on timeout or interruption."""
    start = time.perf_counter()
    with log.open("w") as stream:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            result = process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass
            raise
    if result:
        logs = [log] + sorted(log.parent.glob("rank-*.log"))
        tail = "\n".join(
            (
                f"{p.name}:\n"
                + "\n".join(p.read_text(errors="replace").splitlines()[-12:])
                for p in logs
            )
        )
        raise RuntimeError(f"Job exited {result}; logs at {log.parent}\n{tail}")
    return time.perf_counter() - start


def checkpoint_rows(job, records):
    """Summarize only timed C++ step checkpoints, excluding warmup."""
    rows = []
    for record in records:
        samples = []
        for line in (job / f"rank-{record['rank']}.log").read_text().splitlines():
            if line.startswith("step-profile "):
                samples.append(
                    {
                        k: float(v)
                        for k, v in re.findall("([\\w-]+)=([\\d.eE+-]+)", line)
                    }
                )
        expected = record["iterations"] * len(record["batches"])
        if len(samples) < expected:
            raise RuntimeError(f"Missing C++ step checkpoints in {job}")
        samples = samples[-expected:]
        for stage in samples[0]:
            if stage in ("rank", "t", "attempts"):
                continue
            values = [sample[stage] for sample in samples]
            rows.append(
                {
                    "grid": label(record["shape"]),
                    "mode": record["mode"],
                    "rank": record["rank"],
                    "stage": stage,
                    "samples": len(values),
                    "median_ms": statistics.median(values),
                    "mean_ms": statistics.mean(values),
                    "min_ms": min(values),
                    "max_ms": max(values),
                }
            )
    return rows


def remove_fields(job):
    """Delete this job's verification arrays after comparison."""
    for field in job.glob("field-rank-*.npy"):
        field.unlink()


def write_csv(path, rows):
    """Save the accumulated timing data."""
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def command_output(command, cwd=None):
    """Capture optional machine and revision information."""
    try:
        return subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, timeout=20
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return str(error)


def build_for(args, mode):
    """Choose the source and build belonging to a baseline."""
    repo = args.original_repo if mode == "original" else args.repo
    build = args.original_build_dir if mode == "original" else args.build_dir
    return repo.resolve(), (repo / build).resolve()


def run(args, root, shape, kind, mode, phase):
    """Launch one isolated mode and collect its rank records."""
    ranks, multi, prune = MODES[mode]
    repo, build = build_for(args, mode)
    name = "_mumaxpluscpp_" + args.precision.lower()
    binaries = list((build / "src/bindings").glob(name + "*.so"))
    if len(binaries) != 1:
        raise RuntimeError(f"Expected one {name} backend in {build}/src/bindings")
    job = root / phase / kind / label(shape) / mode
    job.mkdir(parents=True)
    command = [
        sys.executable,
        str(ROOT),
        "--worker",
        "--repo",
        str(repo),
        "--build-dir",
        str(build),
        "--shape",
        label(shape),
        "--kind",
        kind,
        "--mode",
        mode,
        "--phase",
        phase,
        "--job-dir",
        str(job),
        "--precision",
        args.precision,
        "--backend-sha256",
        sha256(binaries[0]),
        "--warmup",
        str(args.warmup),
        "--batches",
        str(args.batches if phase == "timing" else 1),
        "--iterations",
        str(args.iterations if phase == "timing" else 3),
    ]
    if multi:
        command.append("--multigpu")
    if phase == "nsys":
        report = job / ("rank-%q{OMPI_COMM_WORLD_RANK}" if multi else "rank-0")
        command = [
            args.nsys_bin,
            "profile",
            "--trace=cuda,nvtx,mpi",
            "--mpi-impl=openmpi",
            "--sample=none",
            "--cpuctxsw=none",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--output=" + str(report),
        ] + command
    if multi:
        prefix = [args.mpirun, "--bind-to", "none", "-n", str(ranks)]
        if os.geteuid() == 0:
            prefix.append("--allow-run-as-root")
        command = prefix + command
    env = os.environ.copy()
    env.update(
        MUMAXPLUS_FP_PRECISION=args.precision,
        MUMAXPLUS_PROFILE_STEPS="1" if phase == "checkpoints" else "0",
        MUMAXPLUS_PRUNE_DEMAG="1" if prune else "0",
        NVSHMEM_SYMMETRIC_SIZE=args.heap,
    )
    if phase == "nsys":
        env.setdefault("NVSHMEM_NVTX", "common")
    write_json(
        job / "launch.json",
        {
            "command": command,
            "repo": str(repo),
            "commit": command_output(["git", "rev-parse", "HEAD"], repo),
            "dirty": command_output(["git", "status", "--short"], repo),
            "backend_sha256": sha256(binaries[0]),
        },
    )
    print(f"{phase}: {kind} {label(shape)} {mode} ...", flush=True)
    execute(command, env, job / "run.log", args.timeout)
    records = [
        json.loads((job / f"rank-{rank}.json").read_text()) for rank in range(ranks)
    ]
    summary = summarize(records)
    write_json(job / "summary.json", summary)
    if phase == "nsys":
        reports = list(job.glob("*.nsys-rep"))
        if len(reports) != ranks:
            raise RuntimeError(f"Missing Nsight reports in {job}")
        for report in reports:
            execute(
                [
                    args.nsys_bin,
                    "stats",
                    "--report",
                    "cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum",
                    str(report),
                ],
                env,
                report.with_suffix(".stats.txt"),
                args.timeout,
            )
    return job, records, summary


def sweep(args):
    """Compare outputs and collect separate timing and profile data."""
    modes = (["original"] if args.original_repo else []) + [
        "normal",
        "cufftmp1",
        "cufftmp2",
        "pruned2",
    ]
    print(
        f"{len(args.sizes)} grids, {len(modes)} modes, demag + dynamics; "
        f"{args.batches} batches/job."
    )
    print(
        "Original mumax+ baseline: "
        + (str(args.original_repo) if args.original_repo else "NOT configured")
    )
    if args.dry_run:
        return
    if "OMPI_COMM_WORLD_RANK" in os.environ:
        raise RuntimeError("Run this script directly, not through mpirun")
    for tool in [args.mpirun] + ([args.nsys_bin] if args.nsys else []):
        if not shutil.which(tool):
            raise RuntimeError(f"{tool} is missing from the SDK environment")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    print(f"Results: {root}", flush=True)
    write_json(
        root / "metadata.json",
        {
            "settings": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            "python": sys.version,
            "script_sha256": sha256(ROOT),
            "gpu": command_output(["nvidia-smi", "-q"]),
            "topology": command_output(["nvidia-smi", "topo", "-m"]),
            "mpi": command_output([args.mpirun, "--version"]),
            "nsys": command_output([args.nsys_bin, "--version"]) if args.nsys else None,
            "timing": (
                "Median warmed GPU-resident time; setup/downloads excluded. "
                "Single-host rank span."
            ),
            "model": (
                "2 nm cells, Ms=8e5, A=13e-12, alpha=.1, T=0; "
                "bias=(-.0246,.0043,.002) T; fixed Fehlberg dt=1e-13 s."
            ),
            "profiles": "Separate diagnostic jobs; excluded from results.csv.",
        },
    )
    rows, checkpoints = [], []
    jobs = [
        ("timing", kind, shape)
        for kind in ("demag", "dynamics")
        for shape in args.sizes
    ]
    if not args.skip_profiles:
        jobs += [("checkpoints", "dynamics", args.profile_size)]
        if args.nsys:
            jobs += [("nsys", "dynamics", args.profile_size)]
    for phase, kind, shape in jobs:
        selected = (
            [m for m in modes if m != "original"] if phase == "checkpoints" else modes
        )
        reference, summaries, errors = None, {}, {}
        for mode in selected:
            job, records, summary = run(args, root, shape, kind, mode, phase)
            reference = reference or job
            error = compare_fields(reference, job, MODES[mode][0])
            summaries[mode], errors[mode] = summary, error
            print(
                f"  {summary['median_ms']:.4f} ms; comparison PASS, error {error:.3e}",
                flush=True,
            )
            if phase == "checkpoints":
                checkpoints += checkpoint_rows(job, records)
                write_csv(root / "checkpoints.csv", checkpoints)
            if job != reference:
                remove_fields(job)
        if phase == "timing":
            for mode, summary in summaries.items():
                rows.append(
                    {
                        "workload": kind,
                        "grid": label(shape),
                        "mode": mode,
                        "pruning_expected": pruning_expected(shape, mode),
                        **{k: v for k, v in summary.items() if k != "batch_ms"},
                        "speedup_vs_fork_1gpu": summaries["normal"]["median_ms"]
                        / summary["median_ms"],
                        "speedup_vs_original": (
                            summaries["original"]["median_ms"] / summary["median_ms"]
                            if "original" in summaries
                            else ""
                        ),
                        "max_abs_error": errors[mode],
                    }
                )
            write_csv(root / "results.csv", rows)
        remove_fields(reference)
    print(
        f"\nResults: {root}\nTiming: results.csv\nStage breakdown: checkpoints.csv",
        flush=True,
    )


def main():
    """Parse benchmark and private worker arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT.parent.parent)
    parser.add_argument("--build-dir", type=Path, default=Path("build-a100"))
    parser.add_argument("--original-repo", type=Path)
    parser.add_argument("--original-build-dir", type=Path)
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=shape_arg,
        default=[shape_arg(s) for s in DEFAULT_SIZES],
    )
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--precision", choices=("SINGLE", "DOUBLE"), default="SINGLE")
    parser.add_argument("--profile-size", type=shape_arg, default=(512, 512, 32))
    parser.add_argument("--skip-profiles", action="store_true")
    parser.add_argument("--nsys", action="store_true")
    parser.add_argument("--nsys-bin", default="nsys")
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--heap", default="32G")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent
        / "results"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    for name, options in {
        "worker": {"action": "store_true"},
        "shape": {"type": shape_arg},
        "mode": {"choices": MODES},
        "kind": {"choices": ("demag", "dynamics")},
        "phase": {},
        "job-dir": {"type": Path},
        "backend-sha256": {},
        "multigpu": {"action": "store_true"},
    }.items():
        parser.add_argument("--" + name, help=argparse.SUPPRESS, **options)
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    if args.worker:
        args.build_dir = (args.repo / args.build_dir).resolve()
        args.dt = 1e-13
        if any(
            getattr(args, k) is None
            for k in ("shape", "mode", "kind", "phase", "job_dir", "backend_sha256")
        ):
            parser.error("incomplete worker arguments")
        worker(args)
        return
    if bool(args.original_repo) != bool(args.original_build_dir):
        parser.error("provide both --original-repo and --original-build-dir")
    if args.original_repo and args.original_repo.resolve() == args.repo:
        parser.error("the original baseline needs a separate unmodified checkout")
    if (
        args.original_repo
        and build_for(args, "original")[1] == build_for(args, "normal")[1]
    ):
        parser.error("original and fork must use different builds")
    if (
        min(args.batches, args.iterations) < 1
        or args.warmup < 0
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
    ):
        parser.error(
            "positive batches/iterations/timeout and nonnegative warmup required"
        )
    if len(set(args.sizes)) != len(args.sizes):
        parser.error("duplicate sizes")
    if args.quick:
        args.sizes = [(128, 128, 1), (128, 128, 8)]
        args.batches, args.iterations, args.warmup = 2, 5, 3
        args.profile_size = (128, 128, 8)
    sweep(args)


if __name__ == "__main__":
    main()
