#include "demagextrapolation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "constants.hpp"
#include "cudalaunch.hpp"
#include "demag.hpp"
#include "ferromagnet.hpp"
#include "newell.hpp"
#include "stepprofile.hpp"
#include "strayfield.hpp"

namespace {
// Invalid settings are rejected collectively before any rank starts stepping.
double tolerance(const char* name, double fallback) {
  const char* value = std::getenv(name);
  if (!value) return fallback;
  char* end;
  const double result = std::strtod(value, &end);
  return end != value && *end == '\0' && std::isfinite(result) && result >= 0
      ? result : -1;
}

struct History {
  const real* values[6];
  real weights[6];
};

__device__ real3 selfField(CuField m, CuParameter msat, real3 self, int cell) {
  const real ms = msat.valueAt(cell);
  const real3 direction = m.vectorAt(cell);
  return {self.x * ms * direction.x, self.y * ms * direction.y,
          self.z * ms * direction.z};
}

__global__ void k_saveDemagResidual(real* residual, CuField field, CuField m,
                                    CuParameter msat, real3 self,
                                    int first, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const int cell = first + i;
  real3 value{0, 0, 0};
  if (m.cellInGeometry(cell))
    value = field.vectorAt(cell) - selfField(m, msat, self, cell);
  residual[i] = value.x;
  residual[count + i] = value.y;
  residual[2 * count + i] = value.z;
}

__global__ void k_predictDemag(CuField field, CuField m, CuParameter msat,
                               real3 self, History history, int first, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const int cell = first + i;
  if (!m.cellInGeometry(cell)) {
    field.setVectorInCell(cell, real3{0, 0, 0});
    return;
  }
  const real3 local = selfField(m, msat, self, cell);
  for (int c = 0; c < 3; ++c) {
    real value = c == 0 ? local.x : (c == 1 ? local.y : local.z);
    for (int j = 0; j < 6; ++j)
      value += history.weights[j] * history.values[j][c * count + i];
    field.setValueInCell(cell, c, value);
  }
}
// One pair of maxima per block; never read unowned or nonmagnetic cells.
__global__ void k_checkDemag(CuField predicted, CuField exact, real* blocks,
                            int first, int count) {
  __shared__ real errors[BLOCKDIM], scales[BLOCKDIM];
  const int tid = threadIdx.x;
  const int i = blockIdx.x * blockDim.x + tid;
  real error = 0, scale = 0;
  if (i < count && exact.cellInGeometry(first + i)) {
    const real3 value = exact.vectorAt(first + i);
    const real3 delta = predicted.vectorAt(first + i) - value;
    error = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
    scale = value.x * value.x + value.y * value.y + value.z * value.z;
    if (!isfinite(error) || !isfinite(scale)) error = scale = INFINITY;
  }
  errors[tid] = error;
  scales[tid] = scale;
  __syncthreads();
  for (int stride = BLOCKDIM / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      errors[tid] = fmax(errors[tid], errors[tid + stride]);
      scales[tid] = fmax(scales[tid], scales[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    blocks[2 * blockIdx.x] = errors[0];
    blocks[2 * blockIdx.x + 1] = scales[0];
  }
}
}  // namespace

DemagExtrapolator::DemagExtrapolator(const Ferromagnet* magnet, bool localOnly)
    : magnet_(magnet), localOnly_(localOnly),
      slab_(magnet->grid().size().x, magnet->grid().size().y,
            magnet->grid().size().z,
            localOnly ? magnet->world()->session().rank() : 0,
            localOnly ? magnet->world()->session().size() : 1) {
  const char* flag = std::getenv("MUMAXPLUS_DEMAG_CHECK");
  int checked = !(flag && std::strcmp(flag, "0") == 0);
  double tolerances[2] = {tolerance("MUMAXPLUS_DEMAG_ATOL", 1e-6),
                          tolerance("MUMAXPLUS_DEMAG_RTOL", 1e-4)};
  const auto& session = magnet_->world()->session();
  if (session.multiGpuEnabled()) {
    int result = MPI_Allreduce(MPI_IN_PLACE, tolerances, 2, MPI_DOUBLE, MPI_MIN,
                               session.communicator());
    if (result != MPI_SUCCESS) session.abort(result);
    result = MPI_Allreduce(MPI_IN_PLACE, &checked, 1, MPI_INT, MPI_MAX,
                            session.communicator());
    if (result != MPI_SUCCESS) session.abort(result);
  }
  if (tolerances[0] < 0 || tolerances[1] < 0)
    throw std::invalid_argument("Demag tolerances must be finite and nonnegative");
  // Strictest tolerance wins if rank environments differ.
  atol_ = tolerances[0];
  rtol_ = tolerances[1];
  checked_ = checked;
  const real3 cs = magnet->cellsize();
  self_ = {-MU0 * calcNewellNxx({0, 0, 0}, cs),
           -MU0 * calcNewellNyy({0, 0, 0}, cs),
           -MU0 * calcNewellNzz({0, 0, 0}, cs)};
}

void DemagExtrapolator::beginStep(double time) {
  // No duplicate timestamps, backwards history, or samples from old restarts.
  if (time <= previousTime_) samples_ = next_ = 0;
  previousTime_ = startTime_ = time;
  firstStage_ = true;
  exactStep_ = false;
}

Field DemagExtrapolator::exact() {
  if (activeStepTimings) ++stepTimings().demagExact;
  return localOnly_ ? magnet_->getStrayField(magnet_)->evalLocalSlab()
                    : evalDemagField(magnet_);
}

Field DemagExtrapolator::eval(double time) {
  if (firstStage_) {
    // One exact sample at the accepted starting state, never at trial states.
    Field field = exact();
    StepCheckpoint checkpoint;
    const int slot = next_;
    residual_[slot].allocate(size_t(3) * slab_.localCells);
    if (slab_.localCells > 0)
      cudaLaunch(slab_.localCells, k_saveDemagResidual, residual_[slot].get(),
                 field.cu(), magnet_->magnetization()->field().cu(),
                 magnet_->msat.cu(), self_, slab_.firstCell, slab_.localCells);
    times_[slot] = startTime_;
    next_ = (next_ + 1) % 6;
    if (samples_ < 6) ++samples_;
    firstStage_ = false;
    stepTimings().demagHistory += checkpoint();
    return field;
  }
  // Prime with six exact boundary samples before any extrapolation is used.
  if (samples_ < 6 || exactStep_) return exact();

  StepCheckpoint checkpoint;
  History history{};
  for (int j = 0; j < 6; ++j) {
    double weight = 1;
    for (int k = 0; k < 6; ++k)
      if (j != k) weight *= (time - times_[k]) / (times_[j] - times_[k]);
    history.weights[j] = real(weight);
    history.values[j] = residual_[j].get();
  }
  Field field(magnet_->system(), 3);
  if (slab_.localCells > 0)
    cudaLaunch(slab_.localCells, k_predictDemag, field.cu(),
               magnet_->magnetization()->field().cu(), magnet_->msat.cu(),
               self_, history, slab_.firstCell, slab_.localCells);
  if (activeStepTimings) ++stepTimings().demagPredicted;
  stepTimings().demagPredict += checkpoint();
  return field;
}

// This measures field discrepancy at the final candidate state, not an RK
// truncation-error estimate or a bound on all the unchecked stage fields.
bool DemagExtrapolator::checkStep(double time) {
  if (!checked_ || exactStep_ || samples_ < 6) return true;
  StepCheckpoint checkpoint;
  Field predicted = eval(time);
  Field reference = exact();
  const int blocks = slab_.localCells / BLOCKDIM + (slab_.localCells % BLOCKDIM != 0);
  checkBlocks_.allocate(size_t(2) * blocks);
  if (blocks)
    cudaLaunch(slab_.localCells, k_checkDemag, predicted.cu(), reference.cu(),
               checkBlocks_.get(), slab_.firstCell, slab_.localCells);
  const auto maxima = checkBlocks_.getData();
  double squared[2] = {0, 0};
  for (size_t i = 0; i < maxima.size(); i += 2) {
    squared[0] = std::max(squared[0], double(maxima[i]));
    squared[1] = std::max(squared[1], double(maxima[i + 1]));
  }
  const auto& session = magnet_->world()->session();
  if (session.multiGpuEnabled()) {
    const int result = MPI_Allreduce(MPI_IN_PLACE, squared, 2, MPI_DOUBLE, MPI_MAX,
                                     session.communicator());
    if (result != MPI_SUCCESS) session.abort(result);
  }
  const double error = std::sqrt(squared[0]);
  const double limit = atol_ + rtol_ * std::sqrt(squared[1]);
  const double ratio = error == 0 ? 0 : error / limit;
  auto& timings = stepTimings();
  ++timings.demagChecks;
  timings.demagCheckRatio = std::max(timings.demagCheckRatio, ratio);
  timings.demagCheck += checkpoint();
  return std::isfinite(error) && std::isfinite(limit) && error <= limit;
}

void DemagExtrapolator::retryExact() {
  // Discard rejected history. Retry once, at the SAME dt, with exact demag at
  // every stage; the restored starting state seeds a fresh history.
  samples_ = next_ = 0;
  firstStage_ = true;
  exactStep_ = true;
}
