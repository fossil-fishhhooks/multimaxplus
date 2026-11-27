#pragma once

#include <array>

#include "field.hpp"
#include "physicalslab.hpp"

class Ferromagnet;

// Experimental degree-five predictor. Owned by the RK stepper and cleared at
// continuous-advance boundaries. Public field evaluations never use this cache.
class DemagExtrapolator {
 public:
  DemagExtrapolator(const Ferromagnet*, bool localOnly);
  void beginStep(double time);
  Field eval(double time);
  bool checkStep(double time);
  void retryExact();

 private:
  Field exact();
  const Ferromagnet* magnet_;
  bool checked_ = true, exactStep_ = false;
  double atol_ = 1e-6, rtol_ = 1e-4;  // B-field tolerances, in tesla / dimensionless.
  GpuBuffer<real> checkBlocks_;
  bool localOnly_, firstStage_ = true;
  PhysicalSlab slab_;
  real3 self_;  // B_self = self_ * (Ms * m), component by component, in tesla.
  std::array<GpuBuffer<real>, 6> residual_;
  std::array<double, 6> times_{};
  int samples_ = 0, next_ = 0;
  double startTime_ = 0, previousTime_ = -1;
};
