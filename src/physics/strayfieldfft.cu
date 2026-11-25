#include <cufftMp.h>
#include "session.hpp"
#include "world.hpp"
#include <mpi.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <limits>


#include "constants.hpp"
#include "cudalaunch.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "fullmag.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"
#include "magnet.hpp"
#include "parameter.hpp"
#include "quantityevaluator.hpp"
#include "strayfieldfft.hpp"
#include "strayfieldkernel.hpp"
#include "system.hpp"
#include "stepprofile.hpp"

#if FP_PRECISION == SINGLE
const cufftType FFT = CUFFT_R2C;
const cufftType IFFT = CUFFT_C2R;
const auto& fftExec = cufftExecR2C;
const auto& ifftExec = cufftExecC2R;
const cufftType CFFT = CUFFT_C2C;
const auto& cfftExec = cufftExecC2C;
const MPI_Datatype mpiReal = MPI_FLOAT;
#elif FP_PRECISION == DOUBLE
const cufftType FFT = CUFFT_D2Z;
const cufftType IFFT = CUFFT_Z2D;
const auto& fftExec = cufftExecD2Z;
const auto& ifftExec = cufftExecZ2D;
const cufftType CFFT = CUFFT_Z2Z;
const auto& cfftExec = cufftExecZ2Z;
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

// Write this rank's spatial slab straight into the in-place FFT buffers.
// Each row has two extra real slots for even x (one for odd x).
__global__ void k_pad_local(real* mx, real* my, real* mz, Grid paddedGrid,
                            CuField in, CuParameter msat,
                            std::size_t firstInputIdx, std::size_t rowStride,
                            std::size_t localStorageValues) {
  std::size_t localIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (localIdx >= localStorageValues)
    return;

  const std::size_t row = localIdx / rowStride;
  const std::size_t x = localIdx % rowStride;
  real vx = 0, vy = 0, vz = 0;
  if (x < paddedGrid.size().x) {
    const int globalIdx = firstInputIdx + row * paddedGrid.size().x + x;
    const int3 inCoo = paddedGrid.index2coord(globalIdx) - paddedGrid.origin()
                      + in.system.grid.origin();
    if (in.cellInGeometry(inCoo)) {
      const int inIdx = in.system.grid.coord2index(inCoo);
      const real Ms = msat.valueAt(inIdx);
      vx = Ms * in.valueAt(inIdx, 0);
      vy = Ms * in.valueAt(inIdx, 1);
      vz = Ms * in.valueAt(inIdx, 2);
    }
  }
  // Also clear the FFT row-padding slots on every evaluation.
  mx[localIdx] = vx;
  my[localIdx] = vy;
  mz[localIdx] = vz;
}

__device__ void unpadCell(CuField out, CuField in, int outIdx) {

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


__global__ void k_unpad(CuField out, CuField in) {
  unpadCell(out, in, blockIdx.x * blockDim.x + threadIdx.x);
}

// Crop directly from this rank's inverse FFT buffers (pitched real rows).
__global__ void k_unpad_local(CuField out, const real* hx, const real* hy,
                             const real* hz, int3 paddedSize,
                             std::size_t firstInputIdx, std::size_t rowStride,
                             int firstOutputIdx, int outputCount) {
  int localIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (localIdx >= outputCount)
    return;
  const int outIdx = firstOutputIdx + localIdx;
  if (!out.cellInGeometry(outIdx)) {
    out.setVectorInCell(outIdx, real3{0, 0, 0});
    return;
  }

  const Grid outgrid = out.system.grid;
  const int3 outRelCoo = outgrid.index2coord(outIdx) - outgrid.origin();
  const int3 inRelCoo = paddedSize - outgrid.size() + outRelCoo;
  const std::size_t globalIdx =
      inRelCoo.x + paddedSize.x * (inRelCoo.y + paddedSize.y * inRelCoo.z);
  const std::size_t slabIdx = globalIdx - firstInputIdx;
  const std::size_t descriptorIdx =
      (slabIdx / paddedSize.x) * rowStride + slabIdx % paddedSize.x;
  out.setValueInCell(outIdx, 0, hx[descriptorIdx]);
  out.setValueInCell(outIdx, 1, hy[descriptorIdx]);
  out.setValueInCell(outIdx, 2, hz[descriptorIdx]);
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

namespace {

// These are ordinary single-GPU cuFFT plans: no MPI communicator is attached.
// Unlike distributed cuFFTMp plans, they support batched 1D transforms.
struct LocalFFTPlan {
  cufftHandle handle = 0;
  LocalFFTPlan() { checkCufftResult(cufftCreate(&handle)); }
  ~LocalFFTPlan() { if (handle) cufftDestroy(handle); }
  LocalFFTPlan(const LocalFFTPlan&) = delete;
  LocalFFTPlan& operator=(const LocalFFTPlan&) = delete;
  operator cufftHandle() const { return handle; }
};

// Only physical y/z rows are present. Each x row still has the full padded
// transform length and the two real storage slots required by in-place R2C.
__global__ void k_pad_pruned(real* rows, CuField m, CuParameter msat,
                             int3 physical, int firstZ, int rowStride,
                             int componentValues) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= componentValues) return;
  const int x = i % rowStride;
  const int row = i / rowStride;
  real values[3] = {0, 0, 0};
  if (x < physical.x) {
    const int cell = (firstZ * physical.y + row) * physical.x + x;
    if (m.cellInGeometry(cell)) {
      const real ms = msat.valueAt(cell);
      for (int c = 0; c < 3; ++c) values[c] = ms * m.valueAt(cell, c);
    }
  }
  for (int c = 0; c < 3; ++c) rows[c * componentValues + i] = values[c];
}

// Batched matrix transpose. Missing source rows become zeros; columnOffset
// selects the wanted y output on the inverse. Tiling coalesces both sides.
__global__ void k_transpose_pruned(complex* dst, const complex* src,
                                   int rows, int cols, int sourceRows,
                                   int sourceStride, int columnOffset) {
  __shared__ complex tile[32][33];
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  const size_t sourceBase = size_t(blockIdx.z) * sourceRows * sourceStride;
  for (int j = 0; j < 32; j += 8) {
    if (x < cols && y + j < rows) {
      tile[threadIdx.y + j][threadIdx.x] = y + j < sourceRows
          ? src[sourceBase + size_t(y + j) * sourceStride + columnOffset + x]
          : complex{0, 0};
    }
  }
  __syncthreads();
  x = blockIdx.y * 32 + threadIdx.x;
  y = blockIdx.x * 32 + threadIdx.y;
  const size_t destinationBase = size_t(blockIdx.z) * rows * cols;
  for (int j = 0; j < 32; j += 8) {
    if (x < rows && y + j < cols)
      dst[destinationBase + size_t(y + j) * rows + x] =
          tile[threadIdx.x][threadIdx.y + j];
  }
}

void transposePruned(complex* dst, const complex* src, int rows, int cols,
                     int sourceRows, int sourceStride, int columnOffset,
                     int batches = 1) {
  k_transpose_pruned<<<dim3((cols + 31) / 32, (rows + 31) / 32, batches),
                       dim3(32, 8), 0, getCudaStream()>>>(
      dst, src, rows, cols, sourceRows, sourceStride, columnOffset);
  checkCudaError(cudaPeekAtLastError());
}

// Received messages are [source-rank][local-ky][component][local-z][kx].
// Expand only here to the existing [z][local-ky][kx] spectral layout, clearing
// the missing z planes on EVERY evaluation because the z FFT overwrites them.
__global__ void k_unpack_pruned_z(complex* m0, complex* m1, complex* m2,
                                  const complex* received, int nxComplex,
                                  int localY, int localZ, int physicalZ,
                                  int spectrumCells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= spectrumCells) return;
  const int x = i % nxComplex;
  const int y = (i / nxComplex) % localY;
  const int z = i / (nxComplex * localY);
  complex* outputs[3] = {m0, m1, m2};
  for (int c = 0; c < 3; ++c) {
    complex value{0, 0};
    if (z < physicalZ) {
      const size_t source =
          (((size_t(z / localZ) * localY + y) * 3 + c) * localZ
           + z % localZ) * nxComplex + x;
      value = received[source];
    }
    outputs[c][i] = value;
  }
}

// After inverse z, send only the last physicalZ planes. Message order is
// [destination-rank][local-ky][component][local-output-z][kx].
__global__ void k_pack_pruned_z(complex* send, const complex* h0,
                                const complex* h1, const complex* h2,
                                int nxComplex, int localY, int localZ,
                                int cropZ, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  int q = i / nxComplex;
  const int x = i % nxComplex;
  const int z = q % localZ; q /= localZ;
  const int c = q % 3; q /= 3;
  const int y = q % localY;
  const int destination = q / localY;
  const complex* inputs[3] = {h0, h1, h2};
  send[i] = inputs[c][
      (size_t(cropZ + destination * localZ + z) * localY + y)
      * nxComplex + x];
}

__global__ void k_unpad_pruned(CuField out, const real* rows,
                               int physicalX, int firstCell, int rowStride,
                               int cropX, int componentValues, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const int cell = firstCell + i;
  const int source = (i / physicalX) * rowStride + cropX + i % physicalX;
  const bool inside = out.cellInGeometry(cell);
  for (int c = 0; c < 3; ++c)
    out.setValueInCell(cell, c, inside ? rows[c * componentValues + source] : 0);
}

} // namespace

// Balanced physical-z slabs before/after the transform, and the same ky slabs
// as cuFFTMp between them. Tensor spectra and multiplication are unchanged.
struct StrayFieldFFTExecutor::PrunedFFT {
  int3 physical, padded;
  int localZ, localY, nxComplex, rowStride, firstCell;
  int componentValues, localCells, pencilRows, messageValues;
  GpuBuffer<real> xRows;
  GpuBuffer<complex> yRows, send, received;
  LocalFFTPlan xForward, xBackward, yPlan, zPlan;

  PrunedFFT(int3 physical, int3 padded, int rank, int processes)
      : physical(physical), padded(padded), localZ(physical.z / processes),
        localY(padded.y / processes), nxComplex(padded.x / 2 + 1),
        rowStride(2 * nxComplex),
        firstCell(rank * localZ * physical.y * physical.x),
        componentValues(localZ * physical.y * rowStride),
        localCells(localZ * physical.y * physical.x),
        pencilRows(3 * localZ * nxComplex),
        messageValues(2 * localY * pencilRows),
        xRows(size_t(3) * componentValues),
        yRows(size_t(pencilRows) * padded.y),
        send(yRows.size()), received(yRows.size()) {
    size_t worksize;
    int nx[] = {padded.x}, realEmbed[] = {rowStride}, complexEmbed[] = {nxComplex};
    const int xBatch = 3 * localZ * physical.y;
    checkCufftResult(cufftMakePlanMany(xForward, 1, nx,
        realEmbed, 1, rowStride, complexEmbed, 1, nxComplex, FFT, xBatch, &worksize));
    checkCufftResult(cufftMakePlanMany(xBackward, 1, nx,
        complexEmbed, 1, nxComplex, realEmbed, 1, rowStride, IFFT, xBatch, &worksize));
    int ny[] = {padded.y};
    checkCufftResult(cufftMakePlanMany(yPlan, 1, ny,
        ny, 1, padded.y, ny, 1, padded.y, CFFT, pencilRows, &worksize));
    int nz[] = {padded.z};
    const int plane = localY * nxComplex;
    checkCufftResult(cufftMakePlanMany(zPlan, 1, nz,
        nz, plane, 1, nz, plane, 1, CFFT, plane, &worksize));
    for (cufftHandle plan : {xForward.handle, xBackward.handle, yPlan.handle, zPlan.handle})
      checkCufftResult(cufftSetStream(plan, getCudaStream()));
  }
};

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

    // Descriptor allocations and grid ownership remain fixed for this executor.
    for (int comp = 0; comp < 3; comp++) {
      localH_.at(comp) = static_cast<complex*>(hdesc.at(comp)->descriptor->data[0]);
      localM_.at(comp) = static_cast<complex*>(mdesc.at(comp)->descriptor->data[0]);
      inputReal_.at(comp) = static_cast<real*>(mdesc.at(comp)->descriptor->data[0]);
      outputReal_.at(comp) = static_cast<real*>(hdesc.at(comp)->descriptor->data[0]);
    }
    for (int comp = 0; comp < 6; comp++)
      localK_.at(comp) = static_cast<complex*>(kdesc.at(comp)->descriptor->data[0]);

    if (is2D) {
      const int localX = fftSize.x / processes +
                         (session.rank() < fftSize.x % processes ? 1 : 0);
      localSpectrumCells_ = fftSize.y * localX;
    } else {
      const int localY = fftSize.y / processes +
                         (session.rank() < fftSize.y % processes ? 1 : 0);
      localSpectrumCells_ = fftSize.x * localY * fftSize.z;
    }
    use2DKernel_ = is2D && kernel_.grid().origin().z == 0;
    inputRowStride_ = destinationPitch_ / sizeof(real);
    localStorageValues_ = localRows_ * inputRowStride_;

    {
      gatherCounts_.resize(session.size());
      gatherDispls_.resize(session.size());

      const int3 size = kernel_.grid().size();
      const int3 outputSize = system_->grid().size();
      const bool is2D = (size.z == 1);
      const int planes = is2D ? size.y : size.z;
      const int outputPlanes = is2D ? outputSize.y : outputSize.z;
      const int outputValuesPerPlane =
          is2D ? outputSize.x : outputSize.x * outputSize.y;
      const int cropStart = planes - outputPlanes;
      const int base = planes / session.size();
      const int extra = planes % session.size();

      for (int r = 0; r < session.size(); r++) {
        const int planeCount = base + (r < extra ? 1 : 0);
        const int firstPlane = r * base + std::min(r, extra);
        // Clip this rank's padded slab to the physical output interval.
        const int firstOutputPlane =
            std::min(outputPlanes, std::max(0, firstPlane - cropStart));
        const int endOutputPlane = std::min(
            outputPlanes, std::max(0, firstPlane + planeCount - cropStart));
        gatherCounts_.at(r) = (endOutputPlane - firstOutputPlane) * outputValuesPerPlane;
        gatherDispls_.at(r) = firstOutputPlane * outputValuesPerPlane;
      }

      localOutputCount_ = gatherCounts_.at(session.rank());
      firstOutputIdx_ = gatherDispls_.at(session.rank());
    }

    // First version: matching 3D grids, doubled padding, and balanced slabs.
    // Keep the general path for thin films, small odd grids, or uneven splits.
    const int3 physical = system_->grid().size();
    const char* pruneFlag = std::getenv("MUMAXPLUS_PRUNE_DEMAG");
    if ((!pruneFlag || std::strcmp(pruneFlag, "0") != 0) && processes > 1 &&
        magnet_->grid() == system_->grid() &&
        size.x == 2 * physical.x && size.y == 2 * physical.y &&
        size.z == 2 * physical.z && physical.z % processes == 0 &&
        size.y % processes == 0) {
      const size_t z = physical.z / processes;
      const size_t y = size.y / processes;
      const size_t x = fftSize.x;
      const size_t xValues = z * physical.y * 2 * x;
      const size_t exchanged = 3 * z * x * size.y;
      const size_t limit = std::numeric_limits<int>::max();
      if (3 * xValues <= limit && exchanged <= limit &&
          2 * exchanged / processes <= limit &&
          size_t(size.z) * y * x <= limit &&
          3 * z <= 65535 && 3 * z * x <= 32 * 65535 &&
          size.y <= 32 * 65535 && x <= 32 * 65535) {
        pruned_ = std::make_unique<PrunedFFT>(physical, size, session.rank(), processes);
      }
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
  // Finish owning the local plans/buffers before releasing cuFFTMp resources.
  pruned_.reset();
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

Field StrayFieldFFTExecutor::execLocalSlab() const {
  return pruned_ ? execPruned(false) : exec();
}

Field StrayFieldFFTExecutor::execPruned(bool gather) const {
  auto& p = *pruned_;
  const auto& session = magnet_->world()->session();
  StepCheckpoint checkpoint;
  auto& timings = stepTimings();

  auto pad = [&](CuField m, CuParameter ms) {
    cudaLaunchAsync(p.componentValues, k_pad_pruned, p.xRows.get(), m, ms,
                    p.physical, session.rank() * p.localZ, p.rowStride,
                    p.componentValues);
  };
  if (const Ferromagnet* mag = magnet_->asFM()) {
    pad(mag->magnetization()->field().cu(), mag->msat.cu());
  } else {
    auto m = evalHMFullMag(magnet_->asHost());
    auto ms = Parameter(magnet_->system(), 1.0);
    pad(m.cu(), ms.cu());
    // These temporary fields must survive until the pad has consumed them.
    checkCudaError(cudaStreamSynchronize(getCudaStream()));
  }
  timings.demagPad += checkpoint();

  // 1. x FFTs on physical y/z rows, then y FFTs on physical z planes.
  auto xComplex = reinterpret_cast<complex*>(p.xRows.get());
  checkCufftResult(fftExec(p.xForward, p.xRows.get(), xComplex));
  transposePruned(p.yRows.get(), xComplex, p.padded.y, p.nxComplex,
                  p.physical.y, p.nxComplex, 0, 3 * p.localZ);
  checkCufftResult(cfftExec(p.yPlan, p.yRows.get(), p.yRows.get(), CUFFT_FORWARD));

  // 2. Pack by destination ky slab. Only physical z planes travel in MPI.
  transposePruned(p.send.get(), p.yRows.get(), p.pencilRows, p.padded.y,
                  p.pencilRows, p.padded.y, 0);
  auto exchange = [&]() {
    checkCudaError(cudaStreamSynchronize(getCudaStream()));
    const int result = MPI_Alltoall(p.send.get(), p.messageValues, mpiReal,
        p.received.get(), p.messageValues, mpiReal, session.communicator());
    if (result != MPI_SUCCESS) session.abort(result);
  };
  exchange();
  cudaLaunchAsync(localSpectrumCells_, k_unpack_pruned_z,
                  localM_[0], localM_[1], localM_[2],
                  static_cast<const complex*>(p.received.get()),
                  p.nxComplex, p.localY, p.localZ, p.physical.z,
                  localSpectrumCells_);

  // 3. Complete z FFTs in the existing ky-slab layout used by kernel spectra.
  for (int c = 0; c < 3; ++c)
    checkCufftResult(cfftExec(p.zPlan, localM_[c], localM_[c], CUFFT_FORWARD));
  timings.demagForward += checkpoint();

  complex preFactor{-MU0 / kernel_.grid().ncells(), 0};
  cudaLaunchAsync(localSpectrumCells_, k_apply_kernel_3d,
      localH_[0], localH_[1], localH_[2], localM_[0], localM_[1], localM_[2],
      localK_[0], localK_[1], localK_[2], localK_[3], localK_[4], localK_[5],
      preFactor, localSpectrumCells_);
  timings.demagMultiply += checkpoint();

  // 4. Inverse z, then exchange only the planes that survive the z crop.
  for (int c = 0; c < 3; ++c)
    checkCufftResult(cfftExec(p.zPlan, localH_[c], localH_[c], CUFFT_INVERSE));
  cudaLaunchAsync(int(p.send.size()), k_pack_pruned_z,
                  p.send.get(), static_cast<const complex*>(localH_[0]),
                  static_cast<const complex*>(localH_[1]),
                  static_cast<const complex*>(localH_[2]),
                  p.nxComplex, p.localY, p.localZ, p.padded.z - p.physical.z,
                  int(p.send.size()));
  exchange();
  transposePruned(p.yRows.get(), p.received.get(), p.padded.y, p.pencilRows,
                  p.padded.y, p.pencilRows, 0);

  // 5. Inverse y on wanted z planes, inverse x on wanted y/z rows only.
  checkCufftResult(cfftExec(p.yPlan, p.yRows.get(), p.yRows.get(), CUFFT_INVERSE));
  transposePruned(xComplex, p.yRows.get(), p.nxComplex, p.physical.y,
                  p.nxComplex, p.padded.y, p.padded.y - p.physical.y,
                  3 * p.localZ);
  checkCufftResult(ifftExec(p.xBackward, xComplex, p.xRows.get()));
  timings.demagInverse += checkpoint();

  Field h(system_, 3);
  timings.demagSetup += checkpoint();
  cudaLaunchAsync(p.localCells, k_unpad_pruned, h.cu(),
                  static_cast<const real*>(p.xRows.get()),
                  p.physical.x, p.firstCell, p.rowStride,
                  p.padded.x - p.physical.x, p.componentValues, p.localCells);
  checkCudaError(cudaStreamSynchronize(getCudaStream()));
  timings.demagCrop += checkpoint();
  // Public evaluations remain complete; the restricted RK path consumes slabs.
  for (int c = 0; gather && c < 3; ++c) {
    const int result = MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
        h.device_ptr(c), p.localCells, mpiReal, session.communicator());
    if (result != MPI_SUCCESS) session.abort(result);
  }
  timings.demagGather += checkpoint();
  return h;
}

Field StrayFieldFFTExecutor::exec() const {
  if (pruned_) return execPruned();
  StepCheckpoint checkpoint;
  auto& timings = stepTimings();
  
  const Session& session = magnet_->world()->session();

  // Only the normal cuFFT path needs a full padded temporary field.
  std::unique_ptr<Field> mpad;
  if (!session.multiGpuEnabled()) {
    auto kernelSystem = std::make_shared<System>(magnet_->world(), kernel_.grid());
    mpad = std::make_unique<Field>(kernelSystem, 3);
  }

  timings.demagSetup += checkpoint();

  auto padInput = [&](CuField m, CuParameter ms) {
    if (session.multiGpuEnabled()) {
      cudaLaunch(localStorageValues_, k_pad_local,
                 inputReal_.at(0), inputReal_.at(1), inputReal_.at(2),
                 kernel_.grid(), m, ms, sourceOffset_, inputRowStride_,
                 localStorageValues_);
    } else {
      cudaLaunch(mpad->grid().ncells(), k_pad, mpad->cu(), m, ms);
    }
  };

  if (const Ferromagnet* mag = magnet_->asFM()) {
    padInput(mag->magnetization()->field().cu(), mag->msat.cu());
  } else {
    auto hostmag = evalHMFullMag(magnet_->asHost());
    auto ms = Parameter(magnet_->system(), 1.0);
    padInput(hostmag.cu(), ms.cu());
  }
  timings.demagPad += checkpoint();

  if (session.multiGpuEnabled()) {

      for (int comp = 0; comp < 3; comp++) {
        mdesc.at(comp)->subFormat = CUFFT_XT_FORMAT_INPLACE;
        checkCufftResult(cufftXtExecDescriptor(
                        forwardPlan,
                        mdesc.at(comp),
                        mdesc.at(comp),
                        CUFFT_FORWARD));
      }

      timings.demagForward += checkpoint();

      const int ncells = localSpectrumCells_;
      complex preFactor{-MU0 / kernel_.grid().ncells(), 0};
      // Multiplication and inverse FFTs use the same stream, preserving order.
      if (ncells > 0) {
      if (use2DKernel_) {
        // if the h field and m field are two dimensional AND are in the same plane
        // (kernel grid origin at z=0) then the kernel matrix has only 4 relevant
        // components and a more efficient cuda kernel can be used:
        cudaLaunchAsync(ncells, k_apply_kernel_2d, localH_.at(0), localH_.at(1), localH_.at(2),
                   localM_.at(0), localM_.at(1), localM_.at(2), localK_.at(0), localK_.at(1),
                   localK_.at(2), localK_.at(3), preFactor, ncells);
      } else {
        cudaLaunchAsync(ncells, k_apply_kernel_3d, localH_.at(0), localH_.at(1), localH_.at(2),
                   localM_.at(0), localM_.at(1), localM_.at(2), localK_.at(0), localK_.at(1),
                   localK_.at(2), localK_.at(3), localK_.at(4), localK_.at(5), preFactor, ncells);
      }
      }


      timings.demagMultiply += checkpoint();

      for (int comp = 0; comp < 3; comp++) {
        hdesc.at(comp)->subFormat = CUFFT_XT_FORMAT_INPLACE_SHUFFLED;
        checkCufftResult(cufftXtExecDescriptor(backwardPlan, hdesc.at(comp),
                                               hdesc.at(comp), CUFFT_INVERSE));

      }


      timings.demagInverse += checkpoint();

      // Crop locally, then assemble only the physical output on every rank.
      // k_unpad selects the last outputSize cells along each padded axis.
      Field h(system_, 3);
      timings.demagSetup += checkpoint();
      if (localOutputCount_ > 0) {
        cudaLaunch(localOutputCount_, k_unpad_local, h.cu(),
                   static_cast<const real*>(outputReal_.at(0)),
                   static_cast<const real*>(outputReal_.at(1)),
                   static_cast<const real*>(outputReal_.at(2)),
                   kernel_.grid().size(), sourceOffset_, inputRowStride_,
                   firstOutputIdx_, localOutputCount_);
      }
      // Also finish inverse work on ranks with no physical output cells.
      checkCudaError(cudaStreamSynchronize(getCudaStream()));
      timings.demagCrop += checkpoint();

      for (int comp = 0; comp < 3; comp++) {
        int result = MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                                    h.device_ptr(comp), gatherCounts_.data(),
                                    gatherDispls_.data(), mpiReal, session.communicator());
        if (result != MPI_SUCCESS) {
          session.abort(result);
        }
      }
      timings.demagGather += checkpoint();
      return h;
  } else {

    // Forward fourier transforms
    for (int comp = 0; comp < 3; comp++)
      checkCufftResult(fftExec(forwardPlan, mpad->device_ptr(comp), mfft.at(comp)));
    timings.demagForward += checkpoint();

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


    timings.demagMultiply += checkpoint();

    // backward fourier transfrom
    for (int comp = 0; comp < 3; comp++)
      checkCufftResult(ifftExec(backwardPlan, hfft.at(comp), mpad->device_ptr(comp)));
    timings.demagInverse += checkpoint();


    // unpad
    Field h(system_, 3);
    timings.demagSetup += checkpoint();
    cudaLaunch(h.grid().ncells(), k_unpad, h.cu(), mpad->cu());
    timings.demagCrop += checkpoint();
    return h;
  }
}
