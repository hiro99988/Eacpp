#pragma once

#include <mpi.h>

#include "Stopwatches/Stopwatch.hpp"

namespace Eacpp {

class MpiStopwatch : public IStopwatch {
   public:
    MpiStopwatch() {}

    bool IsRunning() const override {
        return isRunning;
    }

    double Elapsed() const override {
        if (isRunning) {
            double now = MPI_Wtime();
            return elapsed + now - start;
        } else {
            return elapsed;
        }
    }

    void Start() override {
        if (!isRunning) {
            start = MPI_Wtime();
            isRunning = true;
        }
    }

    void Stop() override {
        if (isRunning) {
            end = MPI_Wtime();
            isRunning = false;
            elapsed += end - start;
        }
    }

    void Reset() override {
        isRunning = false;
        elapsed = 0.0;
    }

    void Restart() override {
        Reset();
        Start();
    }

   private:
    bool isRunning = false;
    double elapsed = 0.0;
    double start = 0.0;
    double end = 0.0;
};

}  // namespace Eacpp