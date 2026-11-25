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
#include "thermalnoise.hpp"
#include "slabops.hpp"
#include "fieldops.hpp"
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
                            const CuParameter frozenSpins,
                            int firstCell, int localCells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= localCells) return;
  int idx = firstCell + i;

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

namespace {
const Ferromagnet* deterministicMagnet(const DynamicEquation& eq) {
  const auto* rhs = dynamic_cast<const FM_FieldQuantity*>(eq.rhs.get());
  if (!rhs || !(rhs->evaluates(evalTorque) || rhs->evaluates(evalLlgTorque) ||
                rhs->evaluates(evalRelaxTorque))) return nullptr;
  const auto* magnet = rhs->source();
  if (magnet->isSublattice() || eq.x != magnet->magnetization()) return nullptr;
  const auto* noise = dynamic_cast<const FM_FieldQuantity*>(eq.noiseTerm.get());
  const bool zeroNoise = !eq.noiseTerm || eq.noiseTerm->assuredZero() ||
      (noise && noise->source() == magnet && noise->evaluates(evalThermalNoise) &&
       thermalNoiseAssuredZero(magnet));
  return zeroNoise ? magnet : nullptr;
}
}  // namespace

const Ferromagnet* localRkMagnet(const DynamicEquation& eq) {
  const char* flag = std::getenv("MUMAXPLUS_LOCAL_RK");
  if (flag && std::strcmp(flag, "0") == 0) return nullptr;
  const auto* magnet = deterministicMagnet(eq);
  if (!magnet) return nullptr;
  const auto& session = magnet->world()->session();
  if (!session.multiGpuEnabled() || session.size() != 2) return nullptr;
  // Enable the world as a unit: no mixtures with thermal or elastic equations.
  // RK already updates ALL stage variables before evaluating ANY RHS.
  for (const auto& other : magnet->world()->timesolver().equations()) {
    const auto* source = deterministicMagnet(other);
    if (!source || source->world() != magnet->world()) return nullptr;
  }
  return magnet;
}

static Field llgTorque(const Ferromagnet* magnet, bool localOnly) {
  Field torque(magnet->system(), 3);
  const auto size = torque.grid().size();
  const auto& session = magnet->world()->session();
  const PhysicalSlab slab(size.x, size.y, size.z, session.rank(), session.size());
  const int count = localOnly ? slab.localCells : torque.grid().ncells();
  const int first = localOnly ? slab.firstCell : 0;
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
    if (count > 0) cudaLaunch(count, k_llgtorque_fused, torque.cu(), m,
               exchange.cu(), demag.cu(), bias, alpha, gamma, frozenSpins,
               first, count);
    // This includes field assembly as well as the LLG arithmetic.
    timings.llg += checkpoint();
    return torque;
  }
  Field h = localOnly ? evalEffectiveFieldLocalSlab(magnet) : evalEffectiveField(magnet);
  StepCheckpoint checkpoint;
  auto m = magnet->magnetization()->field().cu();
  auto alpha = magnet->alpha.cu();
  auto gamma = magnet->gamma.cu();
  auto frozenSpins = magnet->frozenSpins.cu();
  if (count > 0)
    cudaLaunch(count, k_llgtorque, torque.cu(), m, h.cu(), alpha, gamma, frozenSpins,
               first, count);
  stepTimings().llg += checkpoint();
  return torque;
}

Field evalLlgTorque(const Ferromagnet* magnet) { return llgTorque(magnet, false); }

__global__ void k_dampingtorque(CuField torque,
                                const CuField mField,
                                const CuField hField,
                                const CuParameter gamma,
                                const CuParameter frozenSpins,
                                int firstCell, int localCells) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= localCells) return;
  int idx = firstCell + i;

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

static Field relaxTorque(const Ferromagnet* magnet, bool localOnly) {
  Field torque(magnet->system(), 3);
  Field h = localOnly ? evalEffectiveFieldLocalSlab(magnet) : evalEffectiveField(magnet);
  auto m = magnet->magnetization()->field().cu();
  auto gamma = magnet->gamma.cu();
  auto frozenSpins = magnet->frozenSpins.cu();
  const auto size = torque.grid().size();
  const auto& session = magnet->world()->session();
  const PhysicalSlab slab(size.x, size.y, size.z, session.rank(), session.size());
  const int count = localOnly ? slab.localCells : torque.grid().ncells();
  if (count > 0)
    cudaLaunch(count, k_dampingtorque, torque.cu(), m, h.cu(), gamma, frozenSpins,
               localOnly ? slab.firstCell : 0, count);
  return torque;
}

Field evalRelaxTorque(const Ferromagnet* magnet) { return relaxTorque(magnet, false); }

Field evalLocalTorque(const DynamicEquation& eq) {
  const auto* rhs = static_cast<const FM_FieldQuantity*>(eq.rhs.get());
  const auto* magnet = rhs->source();
  if (rhs->evaluates(evalRelaxTorque)) return relaxTorque(magnet, true);
  Field torque = llgTorque(magnet, true);
  if (rhs->evaluates(evalTorque) && !spinTransferTorqueAssuredZero(magnet)) {
    // STT still evaluates a full field; only its owned cells are consumed.
    const auto size = torque.grid().size();
    const auto& session = magnet->world()->session();
    const PhysicalSlab slab(size.x, size.y, size.z, session.rank(), session.size());
    addToSlab(torque, 1, evalSpinTransferTorque(magnet), slab.firstCell, slab.localCells);
  }
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
