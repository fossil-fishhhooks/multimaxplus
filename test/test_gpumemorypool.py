"""Exercise the actual pool with synthetic large allocations, without CUDA."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_large_buffer_recycling_and_accounting(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    # Each synthetic CUDA allocation reserves one host byte. The real pool must
    # still preserve the requested multi-GiB size in its keys and diagnostics.
    (tmp_path / "cuda_runtime_api.h").write_text(CUDA_STUB)
    source = tmp_path / "check.cpp"
    source.write_text(CHECK)
    root = Path(__file__).resolve().parents[1]
    executable = tmp_path / "check-pool"
    subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-x", "c++",
         "-I", str(tmp_path), "-I", str(root / "src/cudautil"),
         str(root / "src/cudautil/gpumemorypool.cu"), str(source),
         "-o", str(executable)], check=True,
    )
    subprocess.run([str(executable)], check=True)


CUDA_STUB = r'''
#pragma once
#include <cstddef>
#include <cstdlib>
using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0, cudaErrorCudartUnloading = 29;
inline int allocations = 0, releases = 0;
inline cudaError_t cudaMalloc(void** ptr, size_t) {
  *ptr = std::malloc(1);
  ++allocations;
  return *ptr ? cudaSuccess : 1;
}
inline cudaError_t cudaFree(void* ptr) {
  std::free(ptr);
  ++releases;
  return cudaSuccess;
}
inline const char* cudaGetErrorString(cudaError_t) { return "fake CUDA failure"; }
'''

CHECK = r'''
#include <cuda_runtime_api.h>
#include "gpumemorypool.hpp"
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

int main() {
  static_assert(sizeof(size_t) >= 8, "This regression check requires 64-bit sizes");
  const size_t twoGiB = size_t(1) << 31;
  for (size_t bytes : {size_t(64), twoGiB - 1, twoGiB, twoGiB + 13, 2 * twoGiB + 17}) {
    const int beforeAlloc = allocations, beforeFree = releases;
    {
      GpuMemoryPool pool;
      void* ptr = pool.allocate(bytes);
      void* original = ptr;
      pool.recycle(&ptr);
      assert(ptr == nullptr);
      ptr = pool.allocate(bytes);
      assert(ptr == original);
      assert(allocations == beforeAlloc + 1);
      pool.recycle(&ptr);

      std::ostringstream info;
      auto* output = std::cout.rdbuf(info.rdbuf());
      pool.printInfo();
      std::cout.rdbuf(output);
      assert(info.str().find("Total used GPU memory:  " + std::to_string(bytes) +
                             " bytes") != std::string::npos);

      // Live allocations count toward the total as well as cached ones.
      void* extra = pool.allocate(7);
      info.str("");
      output = std::cout.rdbuf(info.rdbuf());
      pool.printInfo();
      std::cout.rdbuf(output);
      assert(info.str().find("Total used GPU memory:  " + std::to_string(bytes + 7) +
                             " bytes") != std::string::npos);
      pool.free(&extra);
      assert(extra == nullptr);
    }
    assert(releases == beforeFree + 2);
  }
}
'''
