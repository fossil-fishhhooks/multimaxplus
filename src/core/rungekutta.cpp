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

RungeKuttaStepper::RungeKuttaStepper(TimeSolver* solver, RKmethod method)
    : butcher_(ButcherTableau::get(method)),
      finalWeightsGpu_(butcher_.weights1) {
  setParentTimeSolver(solver);
  stageWeightsGpu_.reserve(butcher_.rkMatrix.size());
  for (const auto& weights : butcher_.rkMatrix)
    stageWeightsGpu_.emplace_back(weights);
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

  // construct a Runge Kutta stage executor for every equation
  std::vector<RungeKuttaStepper::RungeKuttaStageExecutor> equations;
  for (auto eq : solver_->equations())
    equations.emplace_back(eq, *this);

  timings.setup += checkpoint();
  real t0 = solver_->time();

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
         << " external=" << timings.external << " other-fields=" << timings.otherFields
         << " field-add=" << timings.fieldAdd << " llg=" << timings.llg
         << " error=" << timings.error << " ms\n";
    std::cerr << line.str();
  }
}

RungeKuttaStepper::RungeKuttaStageExecutor::RungeKuttaStageExecutor(
    DynamicEquation eq,
    const RungeKuttaStepper& stepper)
    : x0(eq.x->eval()),
      butcher(stepper.butcher_),
      stepper(stepper),
      x(*eq.x),
      k(stepper.nStages()),
      eq_(eq) {
  localMagnet_ = localRkMagnet(eq);
  if (localMagnet_) {
    stepTimings().localRk = true;
    const auto& session = x.system()->world()->session();
    localCells_ = x0.grid().ncells() / session.size();
    firstCell_ = session.rank() * localCells_;
  }
  // Noise term evaluated only here, it remains constant throughout all stages
  if (eq_.noiseTerm && !eq_.noiseTerm->assuredZero())
    noise = eq_.noiseTerm->eval();
}

void RungeKuttaStepper::RungeKuttaStageExecutor::setStageK(int stage) {
  k[stage] = localMagnet_ ? evalLocalLlgTorque(localMagnet_) : eq_.rhs->eval();

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
                  x.field().cu(), normalize, firstCell_, localCells_);
  gatherStage();

  timings.stageAssign += checkpoint();
}

void RungeKuttaStepper::RungeKuttaStageExecutor::setFinalX() {
  StepCheckpoint checkpoint;
  auto& timings = stepTimings();
  real dt = stepper.solver_->timestep();
  const bool normalize = dynamic_cast<const NormalizedVariable*>(&x) != nullptr;
  addWeightedInto(x0, k, stepper.finalWeightsGpu_, dt, stepper.nStages(),
                  x.field().cu(), normalize, firstCell_, localCells_);
  gatherStage();
  timings.finalAssign += checkpoint();
}

// Keep magnetization replicated for exchange neighbor reads and public APIs.
// k and intermediate fields are valid ONLY on this rank's slab.
void RungeKuttaStepper::RungeKuttaStageExecutor::gatherStage() const {
  if (!localMagnet_) return;
  const auto& session = x.system()->world()->session();
#if FP_PRECISION == SINGLE
  const MPI_Datatype datatype = MPI_FLOAT;
#else
  const MPI_Datatype datatype = MPI_DOUBLE;
#endif
  for (int c = 0; c < 3; ++c) {
    int result = MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
        x.field().device_ptr(c), localCells_, datatype, session.communicator());
    if (result != MPI_SUCCESS) session.abort(result);
  }
}

void RungeKuttaStepper::RungeKuttaStageExecutor::resetX() {
  x = x0;
}

real RungeKuttaStepper::RungeKuttaStageExecutor::getError() const {
  Field err(x.system(), x.ncomp());

  err.makeZero();

  auto dt = stepper.solver_->timestep();

  for (int i = 0; i < stepper.nStages(); i++)
    addTo(err, dt * (butcher.weights1[i] - butcher.weights2[i]), k[i]);

  return maxVecNorm(err);
}
