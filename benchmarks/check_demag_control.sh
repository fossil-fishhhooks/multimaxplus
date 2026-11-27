#!/usr/bin/env bash
# GPU regression suite. Run with bash, outside mpirun; uses the active SDK MPI.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
output=${1:?Usage: bash benchmarks/check_demag_control.sh OUTPUT_DIRECTORY}
build=${BUILD_DIR:-build-a100}
mkdir -p "$output"
# Zero tolerances force a rejected prediction, rollback, and same-dt exact retry.
mpirun --allow-run-as-root --bind-to none -n 2 \
  python benchmarks/demag_extrapolation.py --multigpu --build-dir "$build" \
  --grid 8 6 4 --times-ps 2 --atol 0 --rtol 0 --expect-retry \
  --output "$output/forced-retry"
# Single-GPU check path and thin/uneven ownership layouts.
python benchmarks/demag_extrapolation.py --build-dir "$build" \
  --grid 8 6 4 --times-ps 2 --output "$output/single"
for grid in '17 19 3' '17 19 1'; do
  read -r nx ny nz <<< "$grid"
  mpirun --allow-run-as-root --bind-to none -n 2 \
    python benchmarks/demag_extrapolation.py --multigpu --build-dir "$build" \
    --grid "$nx" "$ny" "$nz" --times-ps 2 \
    --output "$output/ownership-${nx}-${ny}-${nz}"
done
for scenario in reversal domain_wall spin_wave material shaped; do
  mpirun --allow-run-as-root --bind-to none -n 2 \
    python benchmarks/demag_extrapolation.py --multigpu --build-dir "$build" \
    --case "$scenario" --grid 64 32 4 --times-ps 10 100 \
    --output "$output/$scenario"
done
# Different spatial scale, and later trajectory samples.
mpirun --allow-run-as-root --bind-to none -n 2 \
  python benchmarks/demag_extrapolation.py --multigpu --build-dir "$build" \
  --case reversal --grid 64 32 4 --cellsize-nm 3 --times-ps 200 1000 \
  --output "$output/reversal-long"
python - "$output" <<'PY'
import json
from pathlib import Path
import sys
for path in Path(sys.argv[1]).glob('*/rank-1-stats.json'):
    assert json.loads(path.read_text()) == json.loads(
        path.with_name('rank-0-stats.json').read_text()), f'Ranks disagree: {path.parent}'
print('Rank counts/check/retry agreement: PASS')
print('Inspect errors.csv in each case for trajectory accuracy; no global accuracy threshold imposed.')
PY
