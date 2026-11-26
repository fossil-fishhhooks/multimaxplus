#include "rungekutta.hpp"

#include <cmath>
#include <vector>
#include <iostream>
#include <sstream>

#include "butchertableau.hpp"
#include "datatypes.hpp"
#include "dynamicequation.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "fieldquantity.hpp"
#include "reduce.hpp"
#include "timesolver.hpp"
#include "stepprofile.hpp"
#include "variable.hpp"
#include "torque.hpp"
#include "cudaerror.hpp"
#include <mpi.h>
#include "slabops.hpp"

RungeKuttaStepper::RungeKuttaStepper(TimeSolver* solver, RKmethod method)
    : butcher_(ButcherTableau::get(method)), method_(method),
      finalWeightsGpu_(butcher_.weights1) {
  setParentTimeSolver(solver);
  stageWeightsGpu_.reserve(butcher_.rkMatrix.size());
  for (const auto& weights : butcher_.rkMatrix)
    stageWeightsGpu_.emplace_back(weights);
  std::vector<real> errorWeights(butcher_.weights1.size());
  for (size_t i = 0; i < errorWeights.size(); ++i)
    errorWeights[i] = butcher_.weights1[i] - butcher_.weights2[i];
  errorWeightsGpu_ = GpuBuffer<real>(errorWeights);
}

int RungeKuttaStepper::nStages() const {
  return butcher_.nodes.size();
}

void RungeKuttaStepper::step() {
  // If there are no equations to solve, advance time and return early
  if (solver_->equations().empty()) {
    solver_->setTime(solver_->time() + solver_->timestep());
    return;
  }
  StepTimings timings;
  StepTimingScope timingScope(timings);
  StepCheckpoint checkpoint;

  const auto& session = solver_->equations().front().system()->world()->session();
  int localRk = localRkMagnet(solver_->equations().front()) != nullptr;
  if (session.multiGpuEnabled() && session.size() == 2) {
    int result = MPI_Allreduce(MPI_IN_PLACE, &localRk, 1, MPI_INT, MPI_MIN,
                               session.communicator());
    if (result != MPI_SUCCESS) session.abort(result);
  }

  const char* extrapolateFlag = std::getenv("MUMAXPLUS_EXTRAPOLATE_DEMAG");
  int extrapolate = extrapolateFlag && std::strcmp(extrapolateFlag, "1") == 0 &&
      solver_->allowsDemagExtrapolation() && !solver_->hasAdaptiveTimeStep() &&
      solver_->equations().size() == 1 && method_ == RKmethod::FEHLBERG &&
      canExtrapolateDemag(solver_->equations().front());
  if (session.multiGpuEnabled()) {
    int result = MPI_Allreduce(MPI_IN_PLACE, &extrapolate, 1, MPI_INT, MPI_MIN,
                               session.communicator());
    if (result != MPI_SUCCESS) session.abort(result);
  }
  if (!extrapolate) resetDemagHistory();
  else {
    if (!demagPredictor_) {
      const auto* rhs = static_cast<const FM_FieldQuantity*>(solver_->equations().front().rhs.get());
      demagPredictor_ = std::make_unique<DemagExtrapolator>(rhs->source(), localRk != 0);
    }
    demagPredictor_->beginStep(solver_->time());
  }

  // construct a Runge Kutta stage executor for every equation
  std::vector<RungeKuttaStepper::RungeKuttaStageExecutor> equations;
  for (auto eq : solver_->equations())
    equations.emplace_back(eq, *this, localRk != 0);

  timings.setup += checkpoint();
  if (session.multiGpuEnabled() && solver_->hasAdaptiveTimeStep()) {
    // Start every attempt from the same dt, including the first adaptive step.
    real dt = solver_->timestep();
    int result = MPI_Allreduce(MPI_IN_PLACE, &dt, 1, slabRealType(), MPI_MIN,
                               session.communicator());
    if (result != MPI_SUCCESS) session.abort(result);
    solver_->setTimeStep(dt);
  }
  double t0 = solver_->time();

  bool success = false;
  while (!success) {
    ++timings.attempts;
    real dt = solver_->timestep();

    // apply the stages
    for (int stage = 0; stage < nStages(); stage++) {
      solver_->setTime(t0 + dt * butcher_.nodes[stage]);
      for (auto& eq : equations)
        eq.setStageX(stage);
      checkpoint(); // Stage construction is measured separately below.
      for (auto& eq : equations)
        eq.setStageK(stage);
      timings.rhs += checkpoint(); // Inclusive of all field/torque timings.
    }

    // make the actual step
    solver_->setTime(t0 + dt);
    for (auto& eq : equations)
      eq.setFinalX();

    // nothing more to do if time step is fixed
    if (!solver_->hasAdaptiveTimeStep())
      break;

    checkpoint();
    // loop over equations and get the largest error
    real error = 0.0;
    for (auto& eq : equations)
      if (real e = eq.getError(); e > error)
        error = e;

    // All equations and ranks must accept/reject together, including fallbacks.
    if (session.multiGpuEnabled()) {
      int result = MPI_Allreduce(MPI_IN_PLACE, &error, 1, slabRealType(), MPI_MAX,
                                 session.communicator());
      if (result != MPI_SUCCESS) session.abort(result);
    }
    timings.error += checkpoint();
    success = error < solver_->maxError();

    // update the timestep
    real corrFactor;
    if (success) {
      corrFactor = std::pow(solver_->maxError() / error, 1. / (butcher_.order2 + 1));
    } else {
      corrFactor = std::pow(solver_->maxError() / error, 1. / (butcher_.order1 + 1));
    }
    solver_->adaptTimeStep(corrFactor);

    // undo step if not successful
    if (!success) {
      for (auto& eq : equations)
        eq.resetX();
      solver_->setTime(t0);
    }
  }
  if (timingScope.enabled()) {
    const auto& session = solver_->equations().front().x->system()->world()->session();
    std::ostringstream line;
    line << "step-profile rank=" << session.rank()
         << " local-rk=" << timings.localRk
         << " t=" << solver_->time() << " attempts=" << timings.attempts
         << " setup=" << timings.setup
         << " stage-build=" << timings.stageBuild
         << " stage-normalize-assign=" << timings.stageAssign
         << " final-build=" << timings.finalBuild
         << " final-normalize-assign=" << timings.finalAssign
         << " rhs-inclusive=" << timings.rhs
         << " exchange=" << timings.exchange << " demag=" << timings.demag
         << " demag-setup=" << timings.demagSetup << " demag-pad=" << timings.demagPad
         << " demag-forward=" << timings.demagForward
         << " demag-multiply=" << timings.demagMultiply
         << " demag-inverse=" << timings.demagInverse
         << " demag-crop=" << timings.demagCrop << " demag-gather=" << timings.demagGather
         << " demag-exact=" << timings.demagExact
         << " demag-predicted=" << timings.demagPredicted
         << " demag-history=" << timings.demagHistory
         << " demag-predict=" << timings.demagPredict
         << " external=" << timings.external << " other-fields=" << timings.otherFields
         << " field-add=" << timings.fieldAdd << " llg=" << timings.llg
         << " error=" << timings.error << " ms\n";
    std::cerr << line.str();
  }
}

RungeKuttaStepper::RungeKuttaStageExecutor::RungeKuttaStageExecutor(
    DynamicEquation eq,
    const RungeKuttaStepper& stepper, bool localRk)
    : x0(eq.x->eval()),
      butcher(stepper.butcher_),
      stepper(stepper),
      x(*eq.x),
      k(stepper.nStages()),
      eq_(eq) {
  localMagnet_ = localRk ? static_cast<const FM_FieldQuantity*>(eq.rhs.get())->source()
                         : nullptr;
  if (localMagnet_) {
    stepTimings().localRk = true;
    const auto& session = x.system()->world()->session();
    const auto size = x0.grid().size();
    slab_.emplace(size.x, size.y, size.z, session.rank(), session.size());
  }
  // Noise term evaluated only here, it remains constant throughout all stages
  if (!localMagnet_ && eq_.noiseTerm && !eq_.noiseTerm->assuredZero())
    noise = eq_.noiseTerm->eval();
}

void RungeKuttaStepper::RungeKuttaStageExecutor::setStageK(int stage) {
  if (stepper.demagPredictor_)
    k[stage] = evalRkTorque(eq_, localMagnet_ != nullptr, stepper.demagPredictor_.get());
  else k[stage] = localMagnet_ ? evalLocalTorque(eq_) : eq_.rhs->eval();

  auto dt = stepper.solver_->timestep();
  if (noise)
    addTo(k[stage], 1 / sqrt(dt), noise.value());
}

void RungeKuttaStepper::RungeKuttaStageExecutor::setStageX(int stage) {
  if (stage == 0)
    return;

  StepCheckpoint checkpoint;
  auto& timings = stepTimings();
  real dt = stepper.solver_->timestep();
  const bool normalize = dynamic_cast<const NormalizedVariable*>(&x) != nullptr;
  addWeightedInto(x0, k, stepper.stageWeightsGpu_.at(stage), dt, stage,
                  x.field().cu(), normalize,
                  slab_ ? slab_->firstCell : 0, slab_ ? slab_->localCells : -1);
  gatherStage();

  timings.stageAssign += checkpoint();
}

void RungeKuttaStepper::RungeKuttaStageExecutor::setFinalX() {
  StepCheckpoint checkpoint;
  auto& timings = stepTimings();
  real dt = stepper.solver_->timestep();
  const bool normalize = dynamic_cast<const NormalizedVariable*>(&x) != nullptr;
  addWeightedInto(x0, k, stepper.finalWeightsGpu_, dt, stepper.nStages(),
                  x.field().cu(), normalize,
                  slab_ ? slab_->firstCell : 0, slab_ ? slab_->localCells : -1);
  gatherStage();
  timings.finalAssign += checkpoint();
}

// Keep magnetization replicated for exchange neighbor reads and public APIs.
// k and intermediate fields are valid ONLY on this rank's slab.
void RungeKuttaStepper::RungeKuttaStageExecutor::gatherStage() const {
  if (slab_) gatherSlab(x.field(), *slab_, x.system()->world()->session());
}

void RungeKuttaStepper::RungeKuttaStageExecutor::resetX() {
  x = x0;
}

real RungeKuttaStepper::RungeKuttaStageExecutor::getError() const {
  if (slab_) {
    return weightedErrorNorm(k, stepper.errorWeightsGpu_, stepper.solver_->timestep(),
                             slab_->firstCell, slab_->localCells);
  }
  // Preserve the existing behavior for unrelated/custom equation types.
  Field err(x.system(), x.ncomp());
  err.makeZero();
  const real dt = stepper.solver_->timestep();
  for (int i = 0; i < stepper.nStages(); ++i)
    addTo(err, dt * (butcher.weights1[i] - butcher.weights2[i]), k[i]);
  return maxVecNorm(err);
}
