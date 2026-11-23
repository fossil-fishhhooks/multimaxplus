#pragma once

#include <chrono>
#include <cstdlib>
#include <cstring>

#include "cudaerror.hpp"

// Diagnostic totals for one complete solver step, including any retries.
struct StepTimings {
  double setup = 0, stageBuild = 0, stageAssign = 0;
  double finalBuild = 0, finalAssign = 0, rhs = 0, error = 0;
  double exchange = 0, demag = 0, external = 0, otherFields = 0;
  double fieldAdd = 0, llg = 0;
  // Demag substages are included in the outer demag total, not additional work.
  double demagSetup = 0, demagPad = 0, demagForward = 0, demagMultiply = 0;
  double demagInverse = 0, demagCrop = 0, demagGather = 0;
  int attempts = 0;
};

inline thread_local StepTimings* activeStepTimings = nullptr;

inline StepTimings& stepTimings() {
  static thread_local StepTimings unused;
  return activeStepTimings ? *activeStepTimings : unused;
}

// Install a step's totals for nested field evaluators; restore on exit/exception.
class StepTimingScope {
 public:
  explicit StepTimingScope(StepTimings& timings)
      : previous_(activeStepTimings) {
    const char* flag = std::getenv("MUMAXPLUS_PROFILE_STEPS");
    enabled_ = flag && flag[0] && std::strcmp(flag, "0") != 0;
    activeStepTimings = enabled_ ? &timings : nullptr;
  }
  ~StepTimingScope() { activeStepTimings = previous_; }
  bool enabled() const { return enabled_; }
 private:
  StepTimings* previous_;
  bool enabled_;
};

// Same checkpoint pattern as the demag diagnostic. Disabled means no GPU waits.
class StepCheckpoint {
 public:
  StepCheckpoint() : enabled_(activeStepTimings != nullptr) {
    if (enabled_) {
      checkCudaError(cudaDeviceSynchronize());
      previous_ = Clock::now();
    }
  }
  double operator()() {
    if (!enabled_)
      return 0;
    checkCudaError(cudaDeviceSynchronize());
    auto now = Clock::now();
    double ms = std::chrono::duration<double, std::milli>(now - previous_).count();
    previous_ = now;
    return ms;
  }
 private:
  using Clock = std::chrono::steady_clock;
  bool enabled_;
  Clock::time_point previous_;
};
