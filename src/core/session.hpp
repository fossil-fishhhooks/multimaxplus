#pragma once
#include <mpi.h>

/** Lifetime guard for a world's runtime resources.
 *
 * World declares this guard before its other members so construction happens
 * first and destruction happens last, including after derived-class members.
 * This is a per-world scope; process-wide ownership belongs in the hooks.
 */
class Session {
 public:
  Session();
  ~Session();

  // Configure whether subsequently constructed worlds use multiple GPUs. 
  static void setMultiGpuEnabled(bool enabled);

  // Whether this world requested multi-GPU initialization.
  bool multiGpuEnabled() const { return multiGpuEnabled_; }

  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  Session(Session&&) = delete;
  Session& operator=(Session&&) = delete;

  MPI_Comm communicator() const { return comm_; }
  int rank() const { return rank_; }
  int size() const { return nproc_; }

  [[noreturn]] void abort(int errorCode = 1) const noexcept;

 private:
  bool multiGpuEnabled_;
  int rank_ = 0;
  int nproc_ = 1;
  int device_ = -1;
  MPI_Comm comm_ = MPI_COMM_WORLD;
};
