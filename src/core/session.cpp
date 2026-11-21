#include "session.hpp"

#include <cuda_runtime_api.h>
#include <mpi.h>

#include <iostream>
#include <stdexcept>
#include <string>

namespace {
bool multiGpuRequested = false;
}

void Session::setMultiGpuEnabled(bool enabled) {
  multiGpuRequested = enabled;
}

Session::Session() : multiGpuEnabled_(multiGpuRequested) {
  // Leave the normal single-GPU path alone unless --multigpu was requested.
  if (!multiGpuEnabled_) {
    return;
  }

  int fFinalized = 0;
  int fInitialized = 0;
  if (MPI_Finalized(&fFinalized) != MPI_SUCCESS) {
    throw std::runtime_error("Cannot check whether MPI has been finalized.");
  }
  if (fFinalized != 0) {
    throw std::runtime_error(
        "Cannot create a multi-GPU session: MPI has already been finalized.");
  }
  if (MPI_Initialized(&fInitialized) != MPI_SUCCESS) {
    throw std::runtime_error("Cannot check whether MPI has been initialized.");
  }

  if (fInitialized == 0) {
    if (MPI_Init(nullptr, nullptr) != MPI_SUCCESS) {
      throw std::runtime_error("Cannot initialize MPI.");
    }
    ownsMpi_ = true;
  }

  // Both newly initialized and borrowed MPI use the same setup below.
  try {
    if (MPI_Comm_rank(comm_, &rank_) != MPI_SUCCESS) {
      throw std::runtime_error("Cannot get this process's MPI rank.");
    }
    if (MPI_Comm_size(comm_, &nproc_) != MPI_SUCCESS) {
      throw std::runtime_error("Cannot get the MPI process count.");
    }

    int num_gpus = 0;
    cudaError_t error = cudaGetDeviceCount(&num_gpus);
    if (error != cudaSuccess) {
      throw std::runtime_error(std::string("Cannot query GPUs: ") +
                               cudaGetErrorString(error));
    }
    if (num_gpus == 0) {
      throw std::runtime_error("No CUDA GPUs are visible to this process.");
    }
    if (nproc_ > num_gpus) {
      throw std::runtime_error("There are more MPI processes than visible GPUs.");
    }

    // First version: one server, with the same GPU list visible to every rank.
    device_ = rank_;
    error = cudaSetDevice(device_);
    if (error != cudaSuccess) {
      throw std::runtime_error(std::string("Cannot select GPU: ") +
                               cudaGetErrorString(error));
    }
  } catch (const std::exception& error) {
    // A failed constructor has no destructor. Stop the whole MPI job so another
    // rank cannot wait forever for this process to join a distributed operation.
    std::cerr << "Session startup failed on rank " << rank_ << ": "
              << error.what() << std::endl;
    MPI_Abort(comm_, 1);
    throw;
  }
}

Session::~Session() {
  if (!multiGpuEnabled_) {
    return;
  }

  if (ownsMpi_) {
    MPI_Finalize();
  }
}
