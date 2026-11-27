# Checked demag extrapolation

Enable with `MUMAXPLUS_EXTRAPOLATE_DEMAG=1`. Exact demag remains the default.
The predictor is limited to the existing fixed-step Fehlberg, single ordinary
ferromagnet, deterministic fused-field case: exchange, self-demag, static uniform
bias, no PBC, no anisotropy/DMI/magnetoelastic/cross-magnet fields, and static Ms.
History is cleared between `steps(n)` / `run(duration)` calls; callbacks and
adaptive timestepping use exact demag. This change does not add adaptive dt.

Six exact step-boundary samples prime a degree-five residual predictor. The
self-cell term is evaluated from the current magnetization at every stage.
Once primed, every candidate step gets an additional exact demag evaluation at
its final state. Accept the prediction only when

`max_cell |B_pred - B_exact| <= atol + rtol * max_cell |B_exact|`.

Only magnetic cells participate. Slab maxima are reduced across MPI ranks;
all ranks use the same decision. Settings that differ between ranks use the
smallest tolerances, and checking is enabled if any rank enables it.

A failed check restores the starting magnetization and time, discards history,
and repeats the entire step at the SAME dt using exact demag at every stage.
This exact retry is not checked again, preventing a retry loop. The retry's
starting state seeds fresh history. It does not fix an unstable exact-RK dt.

Settings (read when an advance creates its predictor):

| Variable | Default | Meaning |
| --- | --- | --- |
| `MUMAXPLUS_DEMAG_ATOL` | `1e-6` | Absolute B-field tolerance, tesla |
| `MUMAXPLUS_DEMAG_RTOL` | `1e-4` | Relative field tolerance |
| `MUMAXPLUS_DEMAG_CHECK` | `1` | Set `0` for original unchecked prediction |

These are provisional field tolerances, not certified magnetization-error
bounds. The endpoint check cannot bound all intermediate-stage errors or
accumulated trajectory error. Keep using finer exact references to validate a
particular problem. No automatic timestep or check-interval adaptation is added.
The check requires another FFT solve; it is not a free parallel error estimate.

With step profiling enabled, `demag-checks`, `demag-retries`, and
`demag-check-ratio` report checks, exact retries, and the largest error/limit
ratio for that step. A ratio above one rejects the candidate. `demag-check`
is inclusive of its exact/predicted evaluations, not an additional independent
cost to add to the nested demag timings.

## Accuracy and regression coverage

After building, run `bash benchmarks/check_demag_control.sh OUTPUT_DIRECTORY`.
It checks activation, forces rollback/retry with zero tolerances, compares exact
references, exercises single-GPU and two-GPU thin/uneven layouts, samples reversal,
a domain-wall initial state, low-damping spin-wave initial state, two materials,
an elliptical mask, a different cell size, and a longer reversal trajectory.
Rank counts/check/retry agreement is checked after MPI jobs finish. Runtime
assertions test control flow; inspect each `errors.csv` for physical accuracy.
No general accuracy threshold is assumed.

Individual accuracy runs accept `--case`, `--cellsize-nm`, `--atol`, `--rtol`,
`--unchecked`, and `--expect-retry`. `--max-error` optionally limits maximum
magnetization difference against the finest exact reference. The zero-tolerance
case tests fallback even when ordinary single-precision rounding sets the floor.

## Performance

Run `python benchmarks/benchmark_extrapolation.py --output OUTPUT_DIRECTORY`
directly, not through MPI. It launches exact one-GPU, exact two-GPU, unchecked
extrapolation two-GPU, and checked extrapolation two-GPU jobs. `--case`,
`--cellsize-nm`, `--atol`, and `--rtol` select the same model and tolerances.
An untimed profile probe verifies activation/checking. Profiling is disabled
for timed batches; every batch includes predictor startup. CUDA completion and
the single-host span across ranks are included. Setup and state downloads are
excluded. Timed calls do not profile retry counts (that would perturb timing);
run an accuracy/profile job with the same configuration to inspect those.
