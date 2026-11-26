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


def advance(solver, count):
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
    return exact, predicted



def make_case(shape, case="relaxation"):
    """Construct a deterministic supported FM; no callbacks or time-dependent terms."""
    from mumaxplus import Ferromagnet, Grid, World

    nx, ny, nz = shape
    cellsize = (2e-9, 2e-9, 2e-9)
    world = World(cellsize)
    magnet = Ferromagnet(world, Grid(tuple(shape)))
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
    parser.add_argument("--case", choices=("relaxation", "reversal"), default="relaxation")
    parser.add_argument("--grid", type=int, nargs=3, default=(32, 32, 8), metavar=("NX", "NY", "NZ"))
    parser.add_argument("--times-ps", type=float, nargs="+", default=(1, 10, 100))
    parser.add_argument("--dt-ps", type=float, default=0.1)
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
    cellsize = (2e-9, 2e-9, 2e-9)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        metadata = dict(grid=args.grid, cellsize=cellsize, times_ps=args.times_ps,
                        base_dt_ps=args.dt_ps, multigpu=args.multigpu,
                        backend=str(_cpp.__file__), method="Fehlberg", msat=8e5,
                        aex=13e-12, alpha=0.1, case=args.case,
                        bias=(-0.15, 0.03, 0.005) if args.case == "reversal" else (-0.0246, 0.0043, 0.002),
                        max_error=args.max_error)
        (args.output / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Snapshots and errors: {args.output}", flush=True)
    rows = []
    failed = False
    for endpoint, time_ps in enumerate(sorted(set(args.times_ps))):
        states = {}
        # Fine reference first; all remaining comparisons use this same state.
        for mode, refinement in [("exact", 4), ("exact", 2), ("exact", 1),
                                 ("extrapolated", 2), ("extrapolated", 1)]:
            enabled = mode == "extrapolated"
            os.environ["MUMAXPLUS_EXTRAPOLATE_DEMAG"] = str(int(enabled))
            world, magnet, solver = make_case(args.grid, args.case)
            solver.timestep = args.dt_ps * 1e-12 / refinement
            count = round(time_ps / args.dt_ps) * refinement
            exact, predicted = advance(solver, count)
            if enabled and (predicted != 5 * (count - 5) or exact + predicted != 6 * count):
                raise RuntimeError(f"rank={rank}: predictor did not activate; rebuild/check eligibility")
            if not enabled and predicted:
                raise RuntimeError("Exact control unexpectedly used prediction")
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
                         predicted_evaluations=predicted)
                mean_m = m.astype(np.float64).mean(axis=(1, 2, 3))
                states[mode, refinement] = m
                comparisons = [("fine_reference", states["exact", 4])]
                if enabled:
                    comparisons.append(("exact_same_dt", states["exact", refinement]))
                for comparison, reference in comparisons:
                    maximum, rms = errors(m, reference, geometry)
                    rows.append(dict(time_ps=time_ps, actual_time_s=actual_time,
                                     mode=mode, dt_ps=args.dt_ps / refinement,
                                     mean_mx=mean_m[0], mean_my=mean_m[1], mean_mz=mean_m[2],
                                     comparison=comparison, max_error=maximum,
                                     rms_error=rms, exact_evaluations=exact,
                                     predicted_evaluations=predicted))
                    if comparison == "fine_reference" and args.max_error is not None:
                        failed |= maximum > args.max_error
                    print(f"{time_ps:g} ps {mode} dt={args.dt_ps / refinement:g} ps "
                          f"vs {comparison}: max={maximum:.3e} rms={rms:.3e} "
                          f"exact/predicted={exact}/{predicted} mean_mx={mean_m[0]:.5f}", flush=True)
                with (args.output / "errors.csv").open("w", newline="") as output:
                    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            del solver, magnet, world
    if rank == 0:
        if failed:
            raise SystemExit("FAIL: requested error limit exceeded; see errors.csv")
        print("State capture and comparisons complete. " +
              ("Error limit PASS." if args.max_error is not None else
               "No accuracy threshold imposed; inspect convergence in errors.csv."))


if __name__ == "__main__":
    main()
