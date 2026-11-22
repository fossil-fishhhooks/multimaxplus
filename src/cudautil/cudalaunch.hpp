#pragma once

#include "cudaerror.hpp"
#include "cudastream.hpp"

const int BLOCKDIM = 256;

// Source: https://devblogs.nvidia.com/cplusplus-11-in-cuda-variadic-templates/
// Enqueue work on the shared stream; the caller handles completion.
template <typename... Arguments>
void cudaLaunchAsync(int N, void (*kernelfunction)(Arguments...), Arguments... args) {
  dim3 blockDims(BLOCKDIM);
  dim3 gridDims((N + blockDims.x - 1) / blockDims.x);
  cudaStream_t s0 = getCudaStream();
  kernelfunction<<<gridDims, blockDims, 0, s0>>>(args...);
  checkCudaError(cudaPeekAtLastError());
}

template <typename... Arguments>
void cudaLaunch(int N, void (*kernelfunction)(Arguments...), Arguments... args) {
  dim3 blockDims(BLOCKDIM);
  dim3 gridDims((N + blockDims.x - 1) / blockDims.x);
  cudaStream_t s0 = getCudaStream();
  kernelfunction<<<gridDims, blockDims, 0, s0>>>(args...);
  checkCudaError(cudaPeekAtLastError());
  checkCudaError(cudaDeviceSynchronize());
}

template <typename... Arguments>
void cudaLaunchReductionKernel(void (*kernelfunction)(Arguments...),
                               Arguments... args) {
  dim3 blockDims(BLOCKDIM);
  dim3 gridDims(1);
  cudaStream_t s0 = getCudaStream();
  kernelfunction<<<gridDims, blockDims, 0, s0>>>(args...);
  checkCudaError(cudaPeekAtLastError());
  checkCudaError(cudaDeviceSynchronize());
}
