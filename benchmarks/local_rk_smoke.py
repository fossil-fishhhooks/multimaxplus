"""Run with: mpirun -n 2 python benchmarks/local_rk_smoke.py --multigpu.

Compare restricted local RK with the replicated path, including fallback grids.
Uses build-a100; both ranks must see the same two GPUs.
"""

import os
from pathlib import Path
import sys

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root / "build-a100/src/bindings")]

from mumaxplus import Ferromagnet, Grid, World

if os.environ.get("OMPI_COMM_WORLD_SIZE") != "2" or "--multigpu" not in sys.argv:
    raise SystemExit("Launch with two MPI ranks and --multigpu")

failures = []
for shape in [(8, 6, 4), (17, 19, 8), (5, 7, 6), (8, 6, 1), (8, 6, 3)]:
    world = World((2e-9, 2e-9, 2e-9))
    magnet = Ferromagnet(world, Grid(shape))
    magnet.msat = 8e5
    magnet.aex = 13e-12
    magnet.alpha = 0.1
    magnet.temperature = 0
    magnet.bias_magnetic_field = (-0.0246, 0.0043, 0.002)
    z, y, x = np.indices((shape[2], shape[1], shape[0]))
    initial = np.stack((1 + 0.1 * np.sin(x), 0.2 + 0.1 * np.cos(y),
                        0.1 + 0.1 * np.sin(z)))
    solver = world.timesolver
    solver.set_method("Fehlberg")
    solver.adaptive_timestep = False
    snapshots = []
    for flag in ("0", "1"):
        os.environ["MUMAXPLUS_LOCAL_RK"] = flag
        solver.time = 0
        solver.timestep = 1e-13
        magnet.magnetization = initial
        solver.steps(12)
        # Public evaluations must still return complete fields on every rank.
        snapshots.append([np.asarray(magnet.magnetization.get()).copy(),
                          np.asarray(magnet.exchange_field.eval()).copy(),
                          np.asarray(magnet.demag_field.eval()).copy()])
    errors = [float(np.max(np.abs(a - b)))
              for a, b in zip(*snapshots)]
    passed = all(np.isfinite(a).all() and np.allclose(a, b, rtol=5e-5, atol=1e-6)
                 for a, b in zip(*snapshots))
    print(f"rank={os.environ['OMPI_COMM_WORLD_RANK']} grid={shape} "
          f"{'PASS' if passed else 'FAIL'} errors(m,exchange,demag)={errors}",
          flush=True)
    if not passed:
        failures.append(shape)
if failures:
    raise SystemExit(f"Failed cases: {failures}")
