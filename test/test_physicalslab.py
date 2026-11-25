"""Check physical ownership without CUDA or MPI hardware."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_physical_slab_coverage(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = tmp_path / "slab.cpp"
    source.write_text(CPP)
    executable = tmp_path / "slab-check"
    include = Path(__file__).resolve().parents[1] / "src/core"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fsanitize=undefined",
            "-I",
            str(include),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
    )
    subprocess.run([str(executable)], check=True)


CPP = r"""
#include "physicalslab.hpp"
#include <cassert>
#include <climits>

int main() {
  // Exact coverage, contiguous plane boundaries, balance and empty ranks.
  for (int x : {1, 5, 17, 512})
    for (int y : {1, 3, 8, 19})
      for (int z : {1, 2, 3, 8, 32})
        for (int p = 1; p <= 8; ++p) {
          int next = 0;
          int plane = z == 1 ? x : x * y;
          int low = INT_MAX, high = 0;
          for (int rank = 0; rank < p; ++rank) {
            PhysicalSlab s(x, y, z, rank, p);
            assert(s.firstCell == next);
            assert(s.localCells == s.counts[rank]);
            assert(s.firstCell == s.displacements[rank]);
            assert(s.firstCell % plane == 0 && s.localCells % plane == 0);
            assert(s.balanced == ((z == 1 ? y : z) % p == 0));
            next += s.localCells;
            low = std::min(low, s.localCells);
            high = std::max(high, s.localCells);
          }
          assert(next == x * y * z);
          assert(high - low <= plane);
        }
  PhysicalSlab film(5, 7, 1, 1, 2);
  assert(film.firstCell == 20 && film.localCells == 15);
  PhysicalSlab odd(5, 7, 3, 1, 2);
  assert(odd.firstCell == 70 && odd.localCells == 35);
  PhysicalSlab empty(5, 1, 1, 1, 2);
  assert(empty.localCells == 0 && empty.firstCell == 5);
  bool overflow = false, invalid = false;
  try { PhysicalSlab s(INT_MAX, INT_MAX, INT_MAX, 0, 2); }
  catch (const std::overflow_error&) { overflow = true; }
  try { PhysicalSlab s(1, 1, 1, 0, 0); }
  catch (const std::invalid_argument&) { invalid = true; }
  assert(overflow && invalid);
}
"""
