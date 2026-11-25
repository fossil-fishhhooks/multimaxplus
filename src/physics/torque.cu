#include <memory>

#include "antiferromagnet.hpp"
#include "anisotropy.hpp"
#include "demag.hpp"
#include "dmi.hpp"
#include "exchange.hpp"
#include "magnetoelasticfield.hpp"
#include "mumaxworld.hpp"
#include "zeeman.hpp"
#include "cudalaunch.hpp"
#include "effectivefield.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "stt.hpp"
#include "timesolver.hpp"
#include "torque.hpp"
#include "stepprofile.hpp"
#include "strayfield.hpp"
#include "dynamicequation.hpp"
#include <cstdlib>
#include <cstring>

Field evalTorque(const Ferromagnet* magnet) {
  Field torque = evalLlgTorque(magnet);
  if (!spinTransferTorqueAssuredZero(magnet))
    torque += evalSpinTransferTorque(magnet);
  return torque;
}

namespace {
// Conservative specialization: all omitted effective-field terms must be zero.
bool canFuseLlgFields(const Ferromagnet* magnet) {
  return !magnet->isSublattice() &&
         anisotropyAssuredZero(magnet) && inhomoDmiAssuredZero(magnet) &&
         magnetoelasticAssuredZero(magnet) &&
         strayFieldsAssuredZero(magnet) &&
         magnet->biasMagneticField.isUniform() &&
         !magnet->biasMagneticField.isDynamic() &&
         !demagFieldAssuredZero(magnet);
}

__device__ real3 llgTorqueValue(real3 m, real3 h, real a, real g) {
  real3 mxh = cross(m, h);
  real3 mxmxh = cross(m, mxh);
  return -g / (1 + a * a) * (mxh + a * mxmxh);
}
}  // namespace

__global__ void k_llgtorque(CuField torque,
                            const CuField mField,
                            const CuField hField,
                            const CuParameter alpha,
                            const CuParameter gamma,
                            const CuParameter frozenSpins) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Don't do anything outside of the grid.
  if (!torque.cellInGrid(idx))
    return;

  // When outside the geometry or frozen, set to zero and return early
  if (!torque.cellInGeometry(idx) || (frozenSpins.valueAt(idx) != 0)) {
    torque.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  real3 m = mField.vectorAt(idx);
  real3 h = hField.vectorAt(idx);
  real a = alpha.valueAt(idx);
  real g = gamma.valueAt(idx);
  torque.setVectorInCell(idx, llgTorqueValue(m, h, a, g));
}

// Assemble the field in thread-local values, then write only the torque.
__global__ void k_llgtorque_fused(CuField torque,
                                 const CuField mField,
                                 const CuField exchange,
                                 const CuField demag,
                                 real3 bias,
                                 const CuParameter alpha,
                                 const CuParameter gamma,
                                 const CuParameter frozenSpins,
                                 int firstCell, int localCells) {
  int localIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (localIdx >= localCells) return;
  int idx = firstCell + localIdx;
  if (!torque.cellInGrid(idx))
    return;
  if (!torque.cellInGeometry(idx) || frozenSpins.valueAt(idx) != 0) {
    torque.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }
  // Match evalEffectiveField's order: exchange, external, then demag.
  real3 h = exchange.vectorAt(idx) + bias;
  h += demag.vectorAt(idx);
  torque.setVectorInCell(idx, llgTorqueValue(mField.vectorAt(idx), h,
                                           alpha.valueAt(idx), gamma.valueAt(idx)));
}

const Ferromagnet* localRkMagnet(const DynamicEquation& eq) {
  const char* flag = std::getenv("MUMAXPLUS_LOCAL_RK");
  if (flag && std::strcmp(flag, "0") == 0) return nullptr;
  const auto* rhs = dynamic_cast<const FM_FieldQuantity*>(eq.rhs.get());
  if (!rhs || !rhs->evaluates(evalTorque)) return nullptr;
  const auto* magnet = rhs->source();
  const auto& solver = magnet->world()->timesolver();
  const auto& session = magnet->world()->session();
  if (!session.multiGpuEnabled() || session.size() != 2 ||
      solver.hasAdaptiveTimeStep() || solver.equations().size() != 1 ||
      eq.x != magnet->magnetization() ||
      (eq.noiseTerm && !eq.noiseTerm->assuredZero()) ||
      !spinTransferTorqueAssuredZero(magnet) || !canFuseLlgFields(magnet) ||
      !magnet->getStrayField(magnet)->supportsLocalSlab()) return nullptr;
  return magnet;
}

static Field llgTorque(const Ferromagnet* magnet, bool localOnly) {
  Field torque(magnet->system(), 3);
  if (canFuseLlgFields(magnet)) {
    StepCheckpoint checkpoint;
    auto& timings = stepTimings();
    Field exchange = localOnly ? evalExchangeFieldLocalSlab(magnet)
                               : evalExchangeField(magnet);
    timings.exchange += checkpoint();
    const real3 bias = magnet->mumaxWorld()->biasMagneticField +
                       magnet->biasMagneticField.getUniformValue();
    timings.external += checkpoint();
    Field demag = localOnly ? magnet->getStrayField(magnet)->evalLocalSlab()
                            : evalDemagField(magnet);
    timings.demag += checkpoint();

    auto m = magnet->magnetization()->field().cu();
    auto alpha = magnet->alpha.cu();
    auto gamma = magnet->gamma.cu();
    auto frozenSpins = magnet->frozenSpins.cu();
    const auto& session = magnet->world()->session();
    const int count = torque.grid().ncells() / (localOnly ? session.size() : 1);
    const int first = localOnly ? session.rank() * count : 0;
    cudaLaunch(count, k_llgtorque_fused, torque.cu(), m,
               exchange.cu(), demag.cu(), bias, alpha, gamma, frozenSpins,
               first, count);
    // This includes field assembly as well as the LLG arithmetic.
    timings.llg += checkpoint();
    return torque;
  }
  Field h = evalEffectiveField(magnet);
  StepCheckpoint checkpoint;
  auto m = magnet->magnetization()->field().cu();
  auto alpha = magnet->alpha.cu();
  auto gamma = magnet->gamma.cu();
  auto frozenSpins = magnet->frozenSpins.cu();
  int ncells = torque.grid().ncells();
  cudaLaunch(ncells, k_llgtorque, torque.cu(), m, h.cu(), alpha, gamma, frozenSpins);
  stepTimings().llg += checkpoint();
  return torque;
}

Field evalLlgTorque(const Ferromagnet* magnet) { return llgTorque(magnet, false); }
Field evalLocalLlgTorque(const Ferromagnet* magnet) { return llgTorque(magnet, true); }

__global__ void k_dampingtorque(CuField torque,
                                const CuField mField,
                                const CuField hField,
                                const CuParameter gamma,
                                const CuParameter frozenSpins) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Don't do anything outside of the grid.
  if (!torque.cellInGrid(idx))
    return;

  // When outside the geometry or frozen, set to zero and return early
  if (!torque.cellInGeometry(idx) || (frozenSpins.valueAt(idx) != 0)) {
    torque.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  real3 m = mField.vectorAt(idx);
  real3 h = hField.vectorAt(idx);
  real g = gamma.valueAt(idx);
  real3 t = -g * cross(m, cross(m, h));
  torque.setVectorInCell(idx, t);
}

Field evalRelaxTorque(const Ferromagnet* magnet) {
  Field torque(magnet->system(), 3);
  Field h = evalEffectiveField(magnet);
  auto m = magnet->magnetization()->field().cu();
  auto gamma = magnet->gamma.cu();
  auto frozenSpins = magnet->frozenSpins.cu();
  int ncells = torque.grid().ncells();
  cudaLaunch(ncells, k_dampingtorque, torque.cu(), m, h.cu(), gamma, frozenSpins);
  return torque;
}

real evalMaxTorque(const Ferromagnet* magnet) {
  return maxVecNorm(evalTorque(magnet));
}

FM_FieldQuantity torqueQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalTorque, 3, "torque", "rad/s");
}

FM_FieldQuantity llgTorqueQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalLlgTorque, 3, "llg_torque", "rad/s");
}

FM_FieldQuantity relaxTorqueQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalRelaxTorque, 3, "damping_torque", "rad/s");
}

FM_ScalarQuantity maxTorqueQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalMaxTorque, "max_torque", "rad/s");
}
