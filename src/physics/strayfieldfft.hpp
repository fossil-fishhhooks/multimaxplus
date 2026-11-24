#pragma once

#include <cufftMp.h>
#include <mpi.h>

#include <array>
#include <memory>
#include <cstddef>
#include <vector>

#include "datatypes.hpp"
#include "strayfield.hpp"
#include "strayfieldkernel.hpp"

class Field;
class System;
class Magnet;

/** A StraFieldFFTExecutor uses the FFT method to compute stray fields. */
class StrayFieldFFTExecutor : public StrayFieldExecutor {
 public:
  /**
   * Construct a StrayFieldFFTExecutor.
   *
   * @param magnet the source of the stray field
   * @param system the system in which to compute the stray field
   */
  StrayFieldFFTExecutor(const Magnet* magnet,
                        std::shared_ptr<const System> system,
                        int order,
                        double eps,
                        double switchingradius);

  /** Destruct the executor. */
  ~StrayFieldFFTExecutor();

  /** Compute and return the stray field. */
  Field exec() const;

  /** Return the computation method which is METHOD_FFT. */
  Method method() const { return StrayFieldExecutor::METHOD_FFT; }

  /** Return the asymptotic computation order. */
  int order() const { return kernel_.order(); }

  /** Return epsilon. The parameter used to determine the analytical error
   * using epsilon * R³/V
   */
  double eps() const { return kernel_.eps(); }

  /** Return the switching radius. */
  double switchingradius() const { return kernel_.switchingradius(); }

  const StrayFieldKernel& kernel() const { return kernel_; }

 private:
  // Optional row-pruned pipeline; the general cuFFTMp path remains available.
  struct PrunedFFT;
  std::unique_ptr<PrunedFFT> pruned_;
  Field execPruned() const;
  StrayFieldKernel kernel_;
  std::size_t sourceOffset_ = 0;
  std::size_t sourcePitch_ = 0;
  std::size_t destinationPitch_ = 0;
  std::size_t localRows_ = 0;
  // Cached distributed layout and non-owning aliases of descriptor storage.
  std::array<complex*, 3> localH_{}, localM_{};
  std::array<complex*, 6> localK_{};
  std::array<real*, 3> inputReal_{}, outputReal_{};
  std::vector<int> gatherCounts_, gatherDispls_;
  std::size_t inputRowStride_ = 0;
  std::size_t localStorageValues_ = 0;
  int localSpectrumCells_ = 0;
  int localOutputCount_ = 0;
  int firstOutputIdx_ = 0;
  bool use2DKernel_ = false;
  int3 fftSize;
  std::vector<complex*> kfft, mfft, hfft;
  std::vector<cudaLibXtDesc*> kdesc, mdesc, hdesc;
  cufftHandle forwardPlan;
  cufftHandle backwardPlan;
};
