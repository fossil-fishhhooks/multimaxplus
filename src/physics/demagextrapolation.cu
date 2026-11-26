#include "demagextrapolation.hpp"

#include "constants.hpp"
#include "cudalaunch.hpp"
#include "demag.hpp"
#include "ferromagnet.hpp"
#include "newell.hpp"
#include "stepprofile.hpp"
#include "strayfield.hpp"

namespace {
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
}  // namespace

DemagExtrapolator::DemagExtrapolator(const Ferromagnet* magnet, bool localOnly)
    : magnet_(magnet), localOnly_(localOnly),
      slab_(magnet->grid().size().x, magnet->grid().size().y,
            magnet->grid().size().z,
            localOnly ? magnet->world()->session().rank() : 0,
            localOnly ? magnet->world()->session().size() : 1) {
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
}

Field DemagExtrapolator::eval(double time) {
  auto exact = [&]() {
    if (activeStepTimings) ++stepTimings().demagExact;
    return localOnly_ ? magnet_->getStrayField(magnet_)->evalLocalSlab()
                      : evalDemagField(magnet_);
  };
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
  if (samples_ < 6) return exact();

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
