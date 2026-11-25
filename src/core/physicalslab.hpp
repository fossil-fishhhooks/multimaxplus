#pragma once

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

// Ownership of PHYSICAL cells, independent of the FFT's padded decomposition.
// 3D: split z planes. Thin films: split y rows. Empty owners are allowed.
struct PhysicalSlab {
  std::vector<int> counts, displacements;
  int firstCell, localCells;
  bool balanced;

  PhysicalSlab(int nx, int ny, int nz, int rank, int processes)
      : counts(processes > 0 ? processes : 0), displacements(counts.size()) {
    if (nx <= 0 || ny <= 0 || nz <= 0 || processes <= 0 ||
        rank < 0 || rank >= processes)
      throw std::invalid_argument("Invalid physical slab dimensions or rank");
    const long long plane = nz == 1 ? nx : 1LL * nx * ny;
    const int planes = nz == 1 ? ny : nz;
    // Check before multiplying; both field indexing and MPI counts use int.
    if (plane > std::numeric_limits<int>::max() / planes)
      throw std::overflow_error("Physical slab exceeds supported cell count");
    const int base = planes / processes, extra = planes % processes;
    balanced = extra == 0;
    for (int r = 0; r < processes; ++r) {
      counts[r] = int(plane * (base + (r < extra)));
      displacements[r] = int(plane * (r * base + std::min(r, extra)));
    }
    firstCell = displacements[rank];
    localCells = counts[rank];
  }
};
