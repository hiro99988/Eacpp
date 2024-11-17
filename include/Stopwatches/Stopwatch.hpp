#include <chrono>

#include "Stopwatches/IStopwatch.hpp"

namespace Eacpp {

class Stopwatch : public IStopwatch {
   public:
    Stopwatch() {}

    bool IsRunning() const override {
        return isRunning;
    }

    double Elapsed() const override {
        return std::chrono::duration<double>(ElapsedNanoseconds()).count();
    }

    void Start() override {
        if (!isRunning) {
            start = std::chrono::system_clock::now();
            isRunning = true;
        }
    }

    void Stop() override {
        if (isRunning) {
            end = std::chrono::system_clock::now();
            isRunning = false;
            elapsed += end - start;
        }
    }

    void Reset() override {
        isRunning = false;
        elapsed = std::chrono::nanoseconds(0);
    }

    void Restart() override {
        Reset();
        Start();
    }

    std::chrono::nanoseconds ElapsedNanoseconds() const {
        if (isRunning) {
            auto now = std::chrono::high_resolution_clock::now();
            return elapsed + now - start;
        } else {
            return elapsed;
        }
    }

   private:
    bool isRunning = false;
    std::chrono::nanoseconds elapsed = std::chrono::nanoseconds(0);
    std::chrono::time_point<std::chrono::system_clock> start;
    std::chrono::time_point<std::chrono::system_clock> end;
};

}  // namespace Eacpp