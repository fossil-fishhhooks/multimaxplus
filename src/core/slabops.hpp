#pragma once

#include "cudaerror.hpp"
#include "cudastream.hpp"
#include "field.hpp"
#include "physicalslab.hpp"
#include "session.hpp"

inline MPI_Datatype slabRealType() {
#if FP_PRECISION == SINGLE
  return MPI_FLOAT;
#else
  return MPI_DOUBLE;
#endif
}

// Every component remains a separate array. All ranks call this, including
// empty owners; balanced layouts retain the existing Allgather fast path.
inline void gatherSlab(const Field& field, const PhysicalSlab& slab,
                       const Session& session) {
  checkCudaError(cudaStreamSynchronize(getCudaStream()));
  for (int c = 0; c < field.ncomp(); ++c) {
    const int result = slab.balanced
        ? MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, field.device_ptr(c),
                        slab.localCells, slabRealType(), session.communicator())
        : MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, field.device_ptr(c),
                         slab.counts.data(), slab.displacements.data(),
                         slabRealType(), session.communicator());
    if (result != MPI_SUCCESS) session.abort(result);
  }
}
