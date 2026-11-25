"""Two-GPU local-RK regression checks, including adaptive rejection.

Run from the checkout: mpirun -n 2 python benchmarks/local_rk_smoke.py --multigpu
Use --build-dir to select another build. No mpi4py dependency is required.
"""

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np


def capture_steps(operation):
    """Capture C++ checkpoint output without interleaved MPI logs."""
    with tempfile.TemporaryFile(mode="w+b") as log:
        saved = os.dup(2)
        try:
            os.dup2(log.fileno(), 2)
            operation()
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        log.seek(0)
        return log.read().decode()


def main():
    """Compare both paths, verify activation, and check rank agreement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default="build-a100")
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/multimax-local-rk-smoke")
    )
    parser.add_argument("--multigpu", action="store_true")
    args = parser.parse_args()
    if os.environ.get("OMPI_COMM_WORLD_SIZE") != "2" or not args.multigpu:
        parser.error("Launch with two MPI ranks and --multigpu")
    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    root = Path(__file__).resolve().parents[1]
    build = Path(args.build_dir)
    if not build.is_absolute():
        build = root / build
    sys.path[:0] = [str(root), str(build / "src/bindings")]
    from mumaxplus import Ferromagnet, Grid, World, _cpp

    if not Path(_cpp.__file__).resolve().is_relative_to(build.resolve()):
        raise RuntimeError(f"Wrong backend: {_cpp.__file__}")
    os.environ["MUMAXPLUS_PROFILE_STEPS"] = "1"
    args.output.mkdir(parents=True, exist_ok=True)
    failures = []
    cases = [
        ("balanced", (8, 6, 4)),
        ("odd_xy", (17, 19, 8)),
        ("small", (5, 7, 6)),
        ("thin_even", (8, 6, 1)),
        ("thin_uneven", (8, 7, 1)),
        ("uneven_z", (8, 6, 3)),
        ("geometry_material_frozen", (8, 7, 3)),
        ("anisotropy_dmi", (8, 6, 4)),
        ("dynamic_bias_stt", (8, 6, 4)),
        ("no_demag", (8, 6, 3)),
        ("periodic_z", (8, 6, 4)),
        ("two_magnets", (8, 6, 4)),
        ("rk_BogackiShampine", (8, 6, 4)),
        ("rk_CashKarp", (8, 6, 4)),
        ("rk_DormandPrince", (8, 6, 4)),
        ("adaptive_reject", (8, 7, 3)),
        ("adaptive_macrospin", (8, 6, 1)),
        ("thermal_fallback", (8, 6, 4)),
        ("elastic_fallback", (8, 6, 4)),
    ]
    for name, shape in cases:
        periodic = name == "periodic_z"
        world = World(
            (2e-9, 2e-9, 2e-9),
            pbc_repetitions=(0, 0, 1) if periodic else (0, 0, 0),
            mastergrid=Grid((0, 0, shape[2])) if periodic else Grid((0, 0, 0)),
        )
        z, y, x = np.indices((shape[2], shape[1], shape[0]))
        geometry = (x + y) % 5 != 0 if name == "geometry_material_frozen" else None
        magnets = [Ferromagnet(world, Grid(shape), geometry=geometry)]
        if name == "two_magnets":
            magnets.append(Ferromagnet(world, Grid(shape, (shape[0] + 4, 0, 0))))
        initial = np.stack(
            (1 + 0.1 * np.sin(x), 0.2 + 0.1 * np.cos(y), 0.1 + 0.1 * np.sin(z))
        )
        for magnet in magnets:
            magnet.msat = 8e5
            magnet.aex = 13e-12
            magnet.alpha = 0.1
            magnet.temperature = 0
            magnet.bias_magnetic_field = (-0.0246, 0.0043, 0.002)
        magnet = magnets[0]
        if name == "geometry_material_frozen":
            magnet.aex = np.where(z < shape[2] // 2, 10e-12, 15e-12)[None, ...]
            magnet.msat = np.where(x < shape[0] // 2, 7e5, 8e5)[None, ...]
            magnet.frozen_spins = (x == 1).astype(float)[None, ...]
        if name == "anisotropy_dmi":
            magnet.ku1 = 2e4
            magnet.anisU = (0, 0, 1)
            magnet.dmi_tensor.set_interfacial_dmi(0.5e-3)
        if name == "dynamic_bias_stt":
            magnet.bias_magnetic_field.add_time_term(
                lambda t: (0.001 * np.sin(t * 1e12), 0, 0)
            )
            magnet.jcur = (1e11, 0, 0)
            magnet.pol = 0.5
            magnet.xi = 0.1
        if name == "no_demag":
            magnet.enable_demag = False
        if name == "adaptive_macrospin":
            magnet.enable_demag = False
            magnet.aex = 0
            magnet.gamma = 1.76e11
            magnet.bias_magnetic_field = (0, 0, 0.02)
            initial = (1, 0, 0)
        if name == "thermal_fallback":
            magnet.temperature = 1
            magnet.thermal_seed = 7
        if name == "elastic_fallback":
            magnet.rho = 7800
            magnet.C11 = 2.86e11
            magnet.C12 = 1.04e11
            magnet.C44 = 0.99e11
            magnet.enable_elastodynamics = True
            magnet.elastic_displacement = (0, 0, 0)
            magnet.elastic_velocity = (0, 0, 0)
        adaptive = name.startswith("adaptive_")
        solver = world.timesolver
        solver.adaptive_timestep = False
        method = (
            name[3:] if name.startswith("rk_") else ("Heun" if adaptive else "Fehlberg")
        )
        solver.set_method(method)
        solver.max_error = 1e-4
        snapshots = []
        for flag in ("0", "1"):
            os.environ["MUMAXPLUS_LOCAL_RK"] = flag
            solver.time = 0
            solver.adaptive_timestep = adaptive
            dt = 2e-10 if name == "adaptive_macrospin" else 2e-12
            solver.timestep = dt * (1 + rank) if adaptive else 1e-13
            for m in magnets:
                m.magnetization = initial
                m.reset_noise_generator()
            duration = dt * 1.5

            def evolve():
                if adaptive:
                    # Unequal initial dt must be reconciled before the first stage.
                    solver.steps(1)
                    solver.run(duration - solver.time)
                else:
                    solver.steps(12)

            text = capture_steps(evolve)
            expected = flag == "1" and name not in (
                "thermal_fallback",
                "elastic_fallback",
            )
            activations = re.findall(r"local-rk=([01])", text)
            attempts = [int(a) for a in re.findall(r"attempts=(\d+)", text)]
            if not activations or any(int(a) != int(expected) for a in activations):
                failures.append((name, flag, "wrong branch", activations))
            if adaptive and (not attempts or max(attempts) < 2):
                failures.append((name, flag, "did not exercise step rejection"))
            state = {"time": np.asarray(solver.time), "dt": np.asarray(solver.timestep)}
            for i, m in enumerate(magnets):
                state[f"m{i}"] = np.asarray(m.magnetization.get()).copy()
                state[f"exchange{i}"] = np.asarray(m.exchange_field.eval()).copy()
                state[f"demag{i}"] = np.asarray(m.demag_field.eval()).copy()
            if adaptive and not np.isclose(
                solver.time, duration, rtol=1e-5, atol=1e-18
            ):
                failures.append((name, flag, "wrong final time", solver.time))
            if name == "adaptive_macrospin":
                omega_t = 1.76e11 * 0.02 * solver.time / 1.01
                exact = np.array(
                    [
                        np.cos(omega_t) / np.cosh(0.1 * omega_t),
                        np.sin(omega_t) / np.cosh(0.1 * omega_t),
                        np.tanh(0.1 * omega_t),
                    ]
                )[:, None, None, None]
                if not np.allclose(state["m0"], exact, rtol=5e-4, atol=5e-4):
                    failures.append((name, flag, "analytic precession mismatch"))
            snapshots.append(state)
            path = args.output / f"{name}-{flag}-rank-{rank}.npz"
            np.savez(path, **state)
            # Every rank writes before entering this collective field evaluation.
            magnet.demag_field.eval()
            if rank == 0:
                other_path = args.output / f"{name}-{flag}-rank-1.npz"
                with np.load(other_path) as other:
                    if any(not np.array_equal(state[k], other[k]) for k in state):
                        failures.append((name, flag, "ranks disagree"))
                path.unlink()
                other_path.unlink()
            print(
                f"rank={rank} {name} flag={flag} local-rk={sorted(set(activations))} "
                f"expected={int(expected)} max-attempts={max(attempts, default=0)}",
                flush=True,
            )
        error = max(
            float(np.max(np.abs(snapshots[0][f"m{i}"] - snapshots[1][f"m{i}"])))
            for i in range(len(magnets))
        )
        for key in snapshots[0]:
            # dt can differ slightly when adaptive rounding changes the error estimate.
            if key == "dt":
                continue
            atol = 2e-5 if adaptive else 1e-6
            if not (
                np.isfinite(snapshots[1][key]).all()
                and np.allclose(
                    snapshots[0][key], snapshots[1][key], rtol=5e-5, atol=atol
                )
            ):
                failures.append((name, key, "enabled/disabled mismatch"))
        print(f"rank={rank} {name} max magnetization error={error:.3e}", flush=True)
    if failures:
        raise SystemExit(f"rank={rank} FAIL: {failures}")
    print(f"rank={rank}: all local RK checks passed", flush=True)


if __name__ == "__main__":
    main()
