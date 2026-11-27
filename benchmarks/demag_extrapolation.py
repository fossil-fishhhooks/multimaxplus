"""Save and compare exact/extrapolated fixed-step Fehlberg trajectories.

Single GPU: python benchmarks/demag_extrapolation.py
Two GPUs: mpirun -n 2 python benchmarks/demag_extrapolation.py --multigpu
Times and dt are in picoseconds. This is an accuracy experiment, not a benchmark.
Each endpoint starts afresh and uses ONE continuous steps(n) call, preserving
predictor history. No post-step callbacks or solver implementation changes.
"""

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def errors(actual, reference, geometry):
    """Maximum/RMS vector error over magnetic cells, accumulated in double."""
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    squared = np.sum(delta * delta, axis=0)[geometry]
    if squared.size == 0 or not np.isfinite(squared).all():
        raise RuntimeError("Empty geometry or nonfinite magnetization difference")
    return float(np.sqrt(squared.max())), float(np.sqrt(squared.mean()))


def advance(solver, count, details=False):
    """Collect activation counters without flooding the terminal with profiles."""
    with tempfile.TemporaryFile(mode="w+b") as log:
        saved = os.dup(2)
        try:
            os.dup2(log.fileno(), 2)
            solver.steps(count)
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        log.seek(0)
        text = log.read().decode(errors="replace")
    exact = sum(map(int, re.findall(r"demag-exact=(\d+)", text)))
    predicted = sum(map(int, re.findall(r"demag-predicted=(\d+)", text)))
    checks = sum(map(int, re.findall(r"demag-checks=(\d+)", text)))
    retries = sum(map(int, re.findall(r"demag-retries=(\d+)", text)))
    ratios = list(map(float, re.findall(r"demag-check-ratio=([^ ]+)", text)))
    if details:
        return dict(exact=exact, predicted=predicted, checks=checks, retries=retries,
                    check_ratio=max(ratios, default=0))
    return exact, predicted



CASES = ("relaxation", "reversal", "domain_wall", "spin_wave", "material", "shaped")


def make_case(shape, case="relaxation", cellsize_nm=2.0):
    """Construct a deterministic supported FM; no callbacks or time-dependent terms."""
    from mumaxplus import Ferromagnet, Grid, World

    nx, ny, nz = shape
    cellsize = (cellsize_nm * 1e-9,) * 3
    world = World(cellsize)
    geometry = None
    if case == "shaped":
        yy, xx = np.ogrid[:ny, :nx]
        ellipse = ((xx + 0.5 - nx / 2) / (nx / 2))**2 + ((yy + 0.5 - ny / 2) / (ny / 2))**2 < 1
        geometry = np.broadcast_to(ellipse, (nz, ny, nx)).copy()
    magnet = Ferromagnet(world, Grid(tuple(shape)), geometry=geometry)
    magnet.msat = 8e5
    magnet.aex = 13e-12
    magnet.alpha = 0.1
    magnet.temperature = 0
    bias = (-0.15, 0.03, 0.005) if case == "reversal" else (-0.0246, 0.0043, 0.002)
    magnet.bias_magnetic_field = bias
    # Broadcasting keeps temporary arrays small on the performance grids.
    x = np.arange(nx)[None, None, :]
    y = np.arange(ny)[None, :, None]
    z = np.arange(nz)[:, None, None]
    initial = np.empty((3, nz, ny, nx))
    if case == "reversal":
        initial[0] = 1
        initial[1] = 0.02 + 0.01 * np.cos(2 * np.pi * y / ny)
        initial[2] = 0.01 + 0.01 * np.sin(2 * np.pi * x / nx)
    else:
        initial[0] = 1 + 0.1 * np.sin(2 * np.pi * x / nx)
        initial[1] = 0.2 + 0.1 * np.cos(2 * np.pi * y / ny)
        initial[2] = 0.1 + 0.1 * np.sin(2 * np.pi * z / nz)
    if case == "domain_wall":
        wall = (x - (nx - 1) / 2) / 4
        initial[0] = np.tanh(wall)
        initial[1] = 1 / np.cosh(wall)
        initial[2] = 0
        magnet.bias_magnetic_field = (0.02, 0.003, 0)
    elif case == "spin_wave":
        initial[0] = 1
        initial[1] = 0.05 * np.sin(8 * np.pi * x / nx)
        initial[2] = 0.02 * np.cos(8 * np.pi * x / nx)
        magnet.alpha = 0.005
        magnet.bias_magnetic_field = (0.1, 0, 0)
    elif case == "material":
        magnet.msat = np.broadcast_to(np.where(x < nx // 2, 6e5, 9e5), (nz, ny, nx))[None, ...].copy()
        magnet.aex = np.broadcast_to(np.where(x < nx // 2, 8e-12, 16e-12), (nz, ny, nx))[None, ...].copy()
        magnet.alpha = 0.03
    magnet.magnetization = initial
    solver = world.timesolver
    solver.set_method("Fehlberg")
    solver.adaptive_timestep = False
    return world, magnet, solver


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--build-dir", default="build-a100")
    parser.add_argument("--multigpu", action="store_true")
    parser.add_argument("--case", choices=CASES, default="relaxation")
    parser.add_argument("--grid", type=int, nargs=3, default=(32, 32, 8), metavar=("NX", "NY", "NZ"))
    parser.add_argument("--times-ps", type=float, nargs="+", default=(1, 10, 100))
    parser.add_argument("--dt-ps", type=float, default=0.1)
    parser.add_argument("--cellsize-nm", type=float, default=2.0)
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute B-field check tolerance in tesla")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative B-field check tolerance")
    parser.add_argument("--unchecked", action="store_true", help="Compare against the original unchecked predictor")
    parser.add_argument("--expect-retry", action="store_true", help="Require retry coverage (use with zero tolerances)")
    parser.add_argument("--max-error", type=float, help="Optional maximum vector-error limit against the fine reference")
    parser.add_argument("--output", type=Path, default=Path("/tmp/demag-extrapolation-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    args = parser.parse_args()
    if min(args.grid) < 1 or not np.isfinite(args.dt_ps) or args.dt_ps <= 0:
        parser.error("Grid dimensions and dt must be positive")
    if args.max_error is not None and (not np.isfinite(args.max_error) or args.max_error < 0):
        parser.error("--max-error must be finite and nonnegative")
    for time in args.times_ps:
        steps = time / args.dt_ps
        if not np.isfinite(steps) or steps < 6 or not np.isclose(steps, round(steps), rtol=0, atol=1e-8):
            parser.error("Each time must be an integer multiple of dt and at least six steps")
    if any(not np.isfinite(v) or v < 0 for v in (args.atol, args.rtol)) or not (0 < args.cellsize_nm < float("inf")):
        parser.error("Tolerances must be finite and nonnegative; cellsize must be positive")
    if args.expect_retry and args.unchecked:
        parser.error("--expect-retry requires checks")
    os.environ["MUMAXPLUS_DEMAG_CHECK"] = "0" if args.unchecked else "1"
    os.environ["MUMAXPLUS_DEMAG_ATOL"] = str(args.atol)
    os.environ["MUMAXPLUS_DEMAG_RTOL"] = str(args.rtol)
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
    if args.multigpu and int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1")) != 2:
        parser.error("Use two MPI ranks with --multigpu")
    if not args.multigpu and int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1")) != 1:
        parser.error("MPI launches need --multigpu")
    root = Path(__file__).resolve().parents[1]
    build = Path(args.build_dir)
    if not build.is_absolute():
        build = root / build
    sys.path[:0] = [str(root), str(build / "src/bindings")]
    from mumaxplus import _cpp

    if not Path(_cpp.__file__).resolve().is_relative_to(build.resolve()):
        raise RuntimeError(f"Wrong backend: {_cpp.__file__}")
    os.environ["MUMAXPLUS_PROFILE_STEPS"] = "1"
    nx, ny, nz = args.grid
    geometry = np.ones((nz, ny, nx), dtype=bool)
    cellsize = (args.cellsize_nm * 1e-9,) * 3
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        metadata = dict(grid=args.grid, cellsize=cellsize, times_ps=args.times_ps,
                        base_dt_ps=args.dt_ps, multigpu=args.multigpu,
                        backend=str(_cpp.__file__), method="Fehlberg", msat=8e5,
                        aex=13e-12, case=args.case,
                        bias={"reversal": (-0.15, 0.03, 0.005), "domain_wall": (0.02, 0.003, 0), "spin_wave": (0.1, 0, 0)}.get(args.case, (-0.0246, 0.0043, 0.002)),
                        max_error=args.max_error, checked=not args.unchecked,
                        atol=args.atol, rtol=args.rtol,
                        materials="two materials" if args.case == "material" else "uniform",
                        alpha=0.005 if args.case == "spin_wave" else (0.03 if args.case == "material" else 0.1))
        (args.output / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Snapshots and errors: {args.output}", flush=True)
    rows = []
    rank_records = []
    failed = False
    for endpoint, time_ps in enumerate(sorted(set(args.times_ps))):
        states = {}
        # Fine reference first; all remaining comparisons use this same state.
        for mode, refinement in [("exact", 4), ("exact", 2), ("exact", 1),
                                 ("extrapolated", 2), ("extrapolated", 1)]:
            enabled = mode == "extrapolated"
            os.environ["MUMAXPLUS_EXTRAPOLATE_DEMAG"] = str(int(enabled))
            world, magnet, solver = make_case(args.grid, args.case, args.cellsize_nm)
            solver.timestep = args.dt_ps * 1e-12 / refinement
            count = round(time_ps / args.dt_ps) * refinement
            stats = advance(solver, count, details=True)
            exact, predicted = stats["exact"], stats["predicted"]
            checks, retries = stats["checks"], stats["retries"]
            expected = 6 * (count + retries) + 2 * checks
            if enabled and (predicted == 0 or exact + predicted != expected or
                            (not args.unchecked and checks == 0)):
                raise RuntimeError(f"rank={rank}: predictor did not activate; rebuild/check eligibility")
            if enabled and args.expect_retry and retries == 0:
                raise RuntimeError("Expected at least one rejected prediction and exact retry")
            rank_records.append(dict(endpoint=endpoint, mode=mode, refinement=refinement,
                                     exact=exact, predicted=predicted, checks=checks, retries=retries))
            if not enabled and predicted:
                raise RuntimeError("Exact control unexpectedly used prediction")
            geometry = np.asarray(magnet.geometry, dtype=bool)
            m = magnet.magnetization.eval()
            if not np.isfinite(m).all():
                raise RuntimeError(f"Nonfinite state: {mode}, {time_ps} ps")
            actual_time = float(solver.time)
            if not np.isclose(actual_time, time_ps * 1e-12, rtol=1e-4, atol=1e-20):
                raise RuntimeError(f"Endpoint mismatch: {actual_time} versus {time_ps * 1e-12}")
            if rank == 0:
                label = f"{endpoint:02d}-{mode}-dt{refinement}"
                np.savez(args.output / f"{label}.npz", m=m, geometry=geometry,
                         time=actual_time, target_time=time_ps * 1e-12,
                         dt=solver.timestep, grid=args.grid, cellsize=cellsize,
                         case=args.case, extrapolated=enabled, exact_evaluations=exact,
                         predicted_evaluations=predicted, checks=checks, retries=retries, check_ratio=stats["check_ratio"])
                mean_m = m.astype(np.float64)[:, geometry].mean(axis=1)
                states[mode, refinement] = m
                comparisons = [("fine_reference", states["exact", 4])]
                if enabled:
                    comparisons.append(("exact_same_dt", states["exact", refinement]))
                for comparison, reference in comparisons:
                    maximum, rms = errors(m, reference, geometry)
                    if args.expect_retry and comparison == "exact_same_dt" and maximum > 2e-6:
                        raise RuntimeError("Zero-tolerance retry diverged from exact same-dt control")
                    rows.append(dict(time_ps=time_ps, actual_time_s=actual_time,
                                     mode=mode, dt_ps=args.dt_ps / refinement,
                                     mean_mx=mean_m[0], mean_my=mean_m[1], mean_mz=mean_m[2],
                                     comparison=comparison, max_error=maximum,
                                     rms_error=rms, exact_evaluations=exact,
                                     predicted_evaluations=predicted, checks=checks, retries=retries, check_ratio=stats["check_ratio"]))
                    if comparison == "fine_reference" and args.max_error is not None:
                        failed |= maximum > args.max_error
                    print(f"{time_ps:g} ps {mode} dt={args.dt_ps / refinement:g} ps "
                          f"vs {comparison}: max={maximum:.3e} rms={rms:.3e} "
                          f"exact/predicted={exact}/{predicted} checks/retries={checks}/{retries} mean_mx={mean_m[0]:.5f}", flush=True)
                with (args.output / "errors.csv").open("w", newline="") as output:
                    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            del solver, magnet, world
    (args.output / f"rank-{rank}-stats.json").write_text(json.dumps(rank_records, indent=2) + "\n")
    if rank == 0:
        if failed:
            raise SystemExit("FAIL: requested error limit exceeded; see errors.csv")
        print("State capture and comparisons complete. " +
              ("Error limit PASS." if args.max_error is not None else
               "No accuracy threshold imposed; inspect convergence in errors.csv."))


if __name__ == "__main__":
    main()
