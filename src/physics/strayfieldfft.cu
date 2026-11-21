#include <cufftMp.h>
#include "session.hpp"
#include "world.hpp"
#include <mpi.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <vector>

#include <chrono>
#include <iostream>

#include "constants.hpp"
#include "cudalaunch.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "fullmag.hpp"
#include "grid.hpp"
#include "magnet.hpp"
#include "parameter.hpp"
#include "quantityevaluator.hpp"
#include "strayfieldfft.hpp"
#include "strayfieldkernel.hpp"
#include "system.hpp"

#if FP_PRECISION == SINGLE
const cufftType FFT = CUFFT_R2C;
const cufftType IFFT = CUFFT_C2R;
const auto& fftExec = cufftExecR2C;
const auto& ifftExec = cufftExecC2R;
const MPI_Datatype mpiReal = MPI_FLOAT;
#elif FP_PRECISION == DOUBLE
const cufftType FFT = CUFFT_D2Z;
const cufftType IFFT = CUFFT_Z2D;
const auto& fftExec = cufftExecD2Z;
const auto& ifftExec = cufftExecZ2D;
const MPI_Datatype mpiReal = MPI_DOUBLE;
#endif

#define __CUDAOP__ inline __device__ __host__

// No simpel operator overloading due to definition of real2.
__CUDAOP__ complex sum(complex a, complex b) {
#if FP_PRECISION == SINGLE
  return cuCaddf(a, b);
#elif FP_PRECISION == DOUBLE
  return cuCadd(a, b);
#endif
}

__CUDAOP__ complex prod(complex a, complex b) {
#if FP_PRECISION == SINGLE
  return cuCmulf(a, b);
#elif FP_PRECISION == DOUBLE
  return cuCmul(a, b);
#endif
}

__global__ void k_pad(CuField out, CuField in, CuParameter msat) {
  int outIdx = blockIdx.x * blockDim.x + threadIdx.x;

  Grid outgrid = out.system.grid;
  Grid ingrid = in.system.grid;

  if (outIdx >= outgrid.ncells())
    return;

  int3 outCoo = outgrid.index2coord(outIdx);
  int3 inCoo = outCoo - outgrid.origin() + ingrid.origin();
  int inIdx = ingrid.coord2index(inCoo);

  if (in.cellInGeometry(inCoo)) {
    real Ms = msat.valueAt(inIdx);
    for (int c = 0; c < out.ncomp; c++)
      out.setValueInCell(outIdx, c, Ms * in.valueAt(inIdx, c));
  } else {
    for (int c = 0; c < out.ncomp; c++)
      out.setValueInCell(outIdx, c, 0.0);
  }
}

__global__ void k_unpad(CuField out, CuField in) {
  int outIdx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry of destiny field, set to zero and return
  // early
  if (!out.cellInGeometry(outIdx)) {
    if (out.cellInGrid(outIdx))
      out.setVectorInCell(outIdx, real3{0, 0, 0});
    return;
  }

  Grid outgrid = out.system.grid;
  Grid ingrid = in.system.grid;

  // Output coordinate relative to the origin of the output grid
  int3 outRelCoo = outgrid.index2coord(outIdx) - outgrid.origin();

  // Input coordinate relative to the origin of the input grid
  int3 inRelCoo = ingrid.size() - outgrid.size() + outRelCoo;

  int inIdx = ingrid.coord2index(inRelCoo + ingrid.origin());

  for (int c = 0; c < out.ncomp; c++) {
    out.setValueInCell(outIdx, c, in.valueAt(inIdx, c));
  }
}

static void checkCufftResult(cufftResult result) {
  if (result != CUFFT_SUCCESS)
    throw std::runtime_error("cufft error in demag convolution");
}

__global__ void k_apply_kernel_3d(complex* hx,
                                  complex* hy,
                                  complex* hz,
                                  complex* mx,
                                  complex* my,
                                  complex* mz,
                                  complex* kxx,
                                  complex* kyy,
                                  complex* kzz,
                                  complex* kxy,
                                  complex* kxz,
                                  complex* kyz,
                                  complex preFactor,
                                  int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  hx[i] =
      prod(preFactor,
           (sum(sum(prod(kxx[i], mx[i]), prod(kxy[i], my[i])), prod(kxz[i], mz[i]))));
  hy[i] =
      prod(preFactor,
           (sum(sum(prod(kxy[i], mx[i]), prod(kyy[i], my[i])), prod(kyz[i], mz[i]))));
  hz[i] =
      prod(preFactor,
           (sum(sum(prod(kxz[i], mx[i]), prod(kyz[i], my[i])), prod(kzz[i], mz[i]))));
}

__global__ void k_apply_kernel_2d(complex* hx,
                                  complex* hy,
                                  complex* hz,
                                  complex* mx,
                                  complex* my,
                                  complex* mz,
                                  complex* kxx,
                                  complex* kyy,
                                  complex* kzz,
                                  complex* kxy,
                                  complex preFactor,
                                  int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  hx[i] = prod(preFactor, (sum(prod(kxx[i], mx[i]), prod(kxy[i], my[i]))));
  hy[i] = prod(preFactor, (sum(prod(kxy[i], mx[i]), prod(kyy[i], my[i]))));
  hz[i] = prod(preFactor, prod(kzz[i], mz[i]));
}

StrayFieldFFTExecutor::StrayFieldFFTExecutor(const Magnet* magnet,
                                             std::shared_ptr<const System> system,
                                             int order,
                                             double eps,
                                             double switchingradius)
    : StrayFieldExecutor(magnet, system),
      kernel_(system->grid(),
              magnet_->grid(),
              magnet->world(),
              order,
              eps,
              switchingradius),
      kfft(6),
      hfft(3),
      mfft(3),
      kdesc(6),
      mdesc(3),
      hdesc(3) {

  const Session& session = magnet_->world()->session();
  MPI_Comm comm = session.communicator();

    int3 size = kernel_.grid().size();
    fftSize = {size.x / 2 + 1, size.y, size.z};
    int ncells = fftSize.x * fftSize.y * fftSize.z;

  if(session.multiGpuEnabled()){

    const bool is2D = (size.z == 1);
    const int processes = session.size();
    // Built-in slabs split the first two plan dimensions across ranks.
    // Validate before creating plans so invalid grids allocate no plan resources.
    if (is2D && (size.y < processes || size.x < processes)) {
      throw std::invalid_argument(
          "cuFFTMp 2D FFT requires grid dimensions y and x to each be at least "
          "the MPI process count.");
    }
    if (!is2D && (size.z < processes || size.y < processes)) {
      throw std::invalid_argument(
          "cuFFTMp 3D FFT requires grid dimensions z and y to each be at least "
          "the MPI process count.");
    }

    // setup cuFFTMp first
    checkCufftResult(cufftCreate(&forwardPlan)); // in cuFFTMp you must make plans in 2 calls
    checkCufftResult(cufftCreate(&backwardPlan));

    int localPlanes;
    int firstPlane, endPlane;
    if(is2D){
      const int rank = session.rank();
      const int processes = session.size();

      const int planes = size.y;

      const int base = planes / processes;
      const int extra = planes % processes;

      localPlanes = base + (rank < extra ? 1 : 0);
      firstPlane = rank * base + std::min(rank, extra);
      endPlane = firstPlane + localPlanes;  // exclusive
    } else {
      const int rank = session.rank();
      const int processes = session.size();

      const int planes = size.z;

      const int base = planes / processes;
      const int extra = planes % processes;

      localPlanes = base + (rank < extra ? 1 : 0);
      firstPlane = rank * base + std::min(rank, extra);
      endPlane = firstPlane + localPlanes;  // exclusive
    }
    const size_t rowLength = size.x;
    const size_t paddedRowLength = 2 * (rowLength / 2 + 1);

    const size_t rowsPerPlane = is2D ? 1 : size.y;
    const size_t localRows = localPlanes * rowsPerPlane;

    // Offset measured in real values, not bytes
    const size_t sourceOffset =
        firstPlane * rowsPerPlane * rowLength;

    // CUDA copy pitches are measured in bytes i think
    const size_t sourcePitch = rowLength * sizeof(real);
    const size_t destinationPitch = paddedRowLength * sizeof(real);
    const size_t localInputBytes = localRows * destinationPitch;
    
    sourceOffset_ = sourceOffset;
    sourcePitch_ = sourcePitch;
    destinationPitch_ = destinationPitch;
    localRows_ = localRows;

    size_t worksize = 0;
    if(is2D){
      checkCufftResult(cufftMpMakePlan2d(forwardPlan, size.y, size.x, FFT, &comm, CUFFT_COMM_MPI, &worksize));
      checkCufftResult(cufftMpMakePlan2d(backwardPlan, size.y, size.x, IFFT, &comm, CUFFT_COMM_MPI, &worksize));
    }
    else {
      checkCufftResult(cufftMpMakePlan3d(forwardPlan, size.z, size.y, size.x, FFT, &comm, CUFFT_COMM_MPI, &worksize));
      checkCufftResult(cufftMpMakePlan3d(backwardPlan, size.z, size.y, size.x, IFFT, &comm, CUFFT_COMM_MPI, &worksize));
    }

    checkCufftResult(cufftSetStream(forwardPlan, getCudaStream()));
    checkCufftResult(cufftSetStream(backwardPlan, getCudaStream()));

    

    for (auto& desc : kdesc)
      checkCufftResult(cufftXtMalloc(forwardPlan, &desc, CUFFT_XT_FORMAT_INPLACE));
    for (auto& desc : mdesc)
      checkCufftResult(cufftXtMalloc(forwardPlan, &desc, CUFFT_XT_FORMAT_INPLACE));
    for (auto& desc : hdesc)
      checkCufftResult(cufftXtMalloc(backwardPlan, &desc, CUFFT_XT_FORMAT_INPLACE_SHUFFLED));

    for (int comp = 0; comp < 6; comp++) {
      // checkCufftResult(cufftXtExecDescriptor(forwardPlan, kfft.at(comp),
      // kfft.at(comp), CUFFT_FORWARD)); // need tp get data in to kfft.at first!

      real* src = kernel_.field().device_ptr(comp) + sourceOffset;
      real* dest = static_cast<real*>(kdesc.at(comp)->descriptor->data[0]);  // set the pointer to the actual data

      const auto stream = getCudaStream();
      checkCudaError(cudaMemcpy2DAsync(dest, destinationPitch, src, sourcePitch,
                                       sourcePitch, localRows, cudaMemcpyDeviceToDevice,
                                       stream));  // actually write

      // FFT TIME!
      checkCufftResult(cufftXtExecDescriptor(forwardPlan, kdesc.at(comp),
                                             kdesc.at(comp), CUFFT_FORWARD));
    }

  } else {


    for (auto& p : kfft)
      cudaMalloc(reinterpret_cast<void**>(&p), ncells * sizeof(complex));
    for (auto& p : mfft)
      cudaMalloc(reinterpret_cast<void**>(&p), ncells * sizeof(complex));
    for (auto& p : hfft)
      cudaMalloc(reinterpret_cast<void**>(&p), ncells * sizeof(complex));

    checkCufftResult(cufftPlan3d(&forwardPlan, size.z, size.y, size.x, FFT));
    checkCufftResult(cufftPlan3d(&backwardPlan, size.z, size.y, size.x, IFFT));

    cufftSetStream(forwardPlan, getCudaStream());
    cufftSetStream(backwardPlan, getCudaStream());

    for (int comp = 0; comp < 6; comp++)
        checkCufftResult(
           fftExec(forwardPlan, kernel_.field().device_ptr(comp), kfft.at(comp)));
  }
}

StrayFieldFFTExecutor::~StrayFieldFFTExecutor() {
  const Session& session = magnet_->world()->session();
  if(session.multiGpuEnabled()){
    for (auto desc : mdesc)
      cufftXtFree(desc);
    for (auto desc : kdesc)
      cufftXtFree(desc);
    for (auto desc : hdesc)
      cufftXtFree(desc);

    checkCufftResult(cufftDestroy(forwardPlan));
    checkCufftResult(cufftDestroy(backwardPlan));
  }
  else {
    for (auto p : mfft)
      cudaFree(p);
    for (auto p : kfft)
      cudaFree(p);
    for (auto p : hfft)
      cudaFree(p);

    checkCufftResult(cufftDestroy(forwardPlan));
    checkCufftResult(cufftDestroy(backwardPlan));
  }
}

Field StrayFieldFFTExecutor::exec() const {
  using Clock = std::chrono::steady_clock;

  checkCudaError(cudaDeviceSynchronize());
  auto previous = Clock::now();

  auto checkpoint = [&]() {
    checkCudaError(cudaDeviceSynchronize());

    auto now = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(now - previous).count();

    previous = now;
    return ms;
  };

  // pad m, and multiply with msat
  std::shared_ptr<System> kernelSystem =
      std::make_shared<System>(magnet_->world(), kernel_.grid());
  std::unique_ptr<Field> mpad(new Field(kernelSystem, 3));

  if (const Ferromagnet* mag = magnet_->asFM()) {
    auto m = mag->magnetization()->field().cu();
    auto ms = mag->msat.cu();
    cudaLaunch(mpad->grid().ncells(), k_pad, mpad->cu(), m, ms);
  } else {
    auto hostmag = evalHMFullMag(magnet_->asHost());
    auto ms = Parameter(magnet_->system(), 1.0);
    cudaLaunch(mpad->grid().ncells(), k_pad, mpad->cu(), hostmag.cu(), ms.cu());
  }


  double padMs = checkpoint();
  const Session& session = magnet_->world()->session();

  if (session.multiGpuEnabled()) {

      std::vector<complex*> localH(3);
      std::vector<complex*> localM(3);
      std::vector<complex*> localK(6);

      for (int comp = 0; comp < 3; comp++) {
        real* src = mpad->device_ptr(comp) + sourceOffset_;
        real* dest =
            static_cast<real*>(mdesc.at(comp)->descriptor->data[0]);

        const auto stream = getCudaStream();
        checkCudaError(cudaMemcpy2DAsync(dest, destinationPitch_, src, sourcePitch_,
                                        sourcePitch_, localRows_, cudaMemcpyDeviceToDevice,
                                        stream));

        mdesc.at(comp)->subFormat = CUFFT_XT_FORMAT_INPLACE;
        checkCufftResult(cufftXtExecDescriptor(
                        forwardPlan,
                        mdesc.at(comp),
                        mdesc.at(comp),
                        CUFFT_FORWARD));
        localH.at(comp) =
                static_cast<complex*>(hdesc.at(comp)->descriptor->data[0]);
        localM.at(comp) =
                static_cast<complex*>(mdesc.at(comp)->descriptor->data[0]);
        localK.at(comp) =
                static_cast<complex*>(kdesc.at(comp)->descriptor->data[0]);
      }
      // extra ones that dont get copied from the loop
      localK.at(3) = static_cast<complex*>(kdesc.at(3)->descriptor->data[0]);
      localK.at(4) = static_cast<complex*>(kdesc.at(4)->descriptor->data[0]);
      localK.at(5) = static_cast<complex*>(kdesc.at(5)->descriptor->data[0]);

      double copyAndForwardMs = checkpoint();

      int ncells = 0;
      // apply kernel on m_fft
      if (kernel_.grid().size().z == 1) {
        const int processes = session.size();
        const int rank = session.rank();

        const int localX =
            fftSize.x / processes +
            (rank < fftSize.x % processes ? 1 : 0);  // maybe, confirm this

        ncells = fftSize.y * localX;
      } else {
        const int processes = session.size();
        const int rank = session.rank();

        const int localY =
            fftSize.y / processes +
            (rank < fftSize.y % processes ? 1 : 0);  // maybe, confirm this

        ncells = fftSize.x * localY * fftSize.z;
      }

      complex preFactor{-MU0 / kernel_.grid().ncells(), 0};
      if (ncells > 0) {
      if (kernel_.grid().size().z == 1 && kernel_.grid().origin().z == 0) {
        // if the h field and m field are two dimensional AND are in the same plane
        // (kernel grid origin at z=0) then the kernel matrix has only 4 relevant
        // components and a more efficient cuda kernel can be used:
        cudaLaunch(ncells, k_apply_kernel_2d, localH.at(0), localH.at(1), localH.at(2),
                   localM.at(0), localM.at(1), localM.at(2), localK.at(0), localK.at(1),
                   localK.at(2), localK.at(3), preFactor, ncells);
      } else {
        cudaLaunch(ncells, k_apply_kernel_3d, localH.at(0), localH.at(1), localH.at(2),
                   localM.at(0), localM.at(1), localM.at(2), localK.at(0), localK.at(1),
                   localK.at(2), localK.at(3), localK.at(4), localK.at(5), preFactor, ncells);
      }
      }

      double multiplyMs = checkpoint();

      for (int comp = 0; comp < 3; comp++) {
        hdesc.at(comp)->subFormat = CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
        checkCufftResult(cufftXtExecDescriptor(backwardPlan, hdesc.at(comp),
                                               hdesc.at(comp), CUFFT_INVERSE));

        real* dest = mpad->device_ptr(comp) + sourceOffset_;
        real* src = static_cast<real*>(hdesc.at(comp)->descriptor->data[0]);

        const auto stream = getCudaStream();
        checkCudaError(cudaMemcpy2DAsync(dest, sourcePitch_, src, destinationPitch_,
                                         sourcePitch_, localRows_,
                                         cudaMemcpyDeviceToDevice, stream));
      }

      double inverseAndCopyMs = checkpoint();

      // reassembly
      std::vector<int> counts(session.size());
      std::vector<int> displs(session.size());

      const int3 size = kernel_.grid().size();
      const bool is2D = (size.z == 1);

      const int planes = is2D ? size.y : size.z;
      const int valuesPerPlane = is2D ? size.x : size.x * size.y;

      const int base = planes / session.size();
      const int extra = planes % session.size();

      for (int r = 0; r < session.size(); r++) {
        const int planeCount = base + (r < extra ? 1 : 0);
        const int firstPlane = r * base + std::min(r, extra);
        counts.at(r) = planeCount * valuesPerPlane;
        displs.at(r) = firstPlane * valuesPerPlane;
      }

      checkCudaError(cudaStreamSynchronize(getCudaStream()));

      for (int comp = 0; comp < 3; comp++) {
        int result = MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                                    mpad->device_ptr(comp), counts.data(),
                                    displs.data(), mpiReal, session.communicator());
        if (result != MPI_SUCCESS) {
          session.abort(result);
        }
      }
      double gatherMs = checkpoint();

      // unpad
      Field h(system_, 3);
      cudaLaunch(h.grid().ncells(), k_unpad, h.cu(), mpad->cu());

      double unpadMs = checkpoint();
      std::cerr << "rank=" << session.rank() << " pad=" << padMs
                << " copy+forward=" << copyAndForwardMs << " multiply=" << multiplyMs
                << " inverse+copy=" << inverseAndCopyMs << " gather=" << gatherMs
                << " unpad=" << unpadMs << " ms\n";
      return h;
  } else {

    // Forward fourier transforms
    for (int comp = 0; comp < 3; comp++)
      checkCufftResult(fftExec(forwardPlan, mpad->device_ptr(comp), mfft.at(comp)));
    double forwardMs = checkpoint();

    // apply kernel on m_fft
    int ncells = fftSize.x * fftSize.y * fftSize.z;
    complex preFactor{-MU0 / kernel_.grid().ncells(), 0};
    if (kernel_.grid().size().z == 1 && kernel_.grid().origin().z == 0) {
      // if the h field and m field are two dimensional AND are in the same plane
      // (kernel grid origin at z=0) then the kernel matrix has only 4 relevant
      // components and a more efficient cuda kernel can be used:
      cudaLaunch(ncells, k_apply_kernel_2d, hfft.at(0), hfft.at(1), hfft.at(2),
                mfft.at(0), mfft.at(1), mfft.at(2), kfft.at(0), kfft.at(1), kfft.at(2),
                kfft.at(3), preFactor, ncells);
    } else {
      cudaLaunch(ncells, k_apply_kernel_3d, hfft.at(0), hfft.at(1), hfft.at(2),
                mfft.at(0), mfft.at(1), mfft.at(2), kfft.at(0), kfft.at(1), kfft.at(2),
                kfft.at(3), kfft.at(4), kfft.at(5), preFactor, ncells);
    }

    double multiplyMs = checkpoint();

    // backward fourier transfrom
    for (int comp = 0; comp < 3; comp++)
      checkCufftResult(ifftExec(backwardPlan, hfft.at(comp), mpad->device_ptr(comp)));
    double inverseMs = checkpoint();

    // unpad
    Field h(system_, 3);
    cudaLaunch(h.grid().ncells(), k_unpad, h.cu(), mpad->cu());
    double unpadMs = checkpoint();
    std::cerr << "mode=normal rank=" << session.rank() << " pad=" << padMs
              << " forward=" << forwardMs << " multiply=" << multiplyMs
              << " inverse=" << inverseMs << " unpad=" << unpadMs << " ms\n";
    return h;
  }
}
