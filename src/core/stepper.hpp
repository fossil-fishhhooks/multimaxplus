#pragma once

class TimeSolver;

class Stepper {
 public:
  virtual void step() = 0;
  virtual ~Stepper() {}
  virtual void resetDemagHistory() {}
  void setParentTimeSolver(TimeSolver*);

 protected:
  TimeSolver* solver_ = nullptr;
};
