#pragma once

namespace eacpp {

struct IStopwatch {
    virtual ~IStopwatch() {}
    virtual bool IsRunning() const = 0;
    virtual double Elapsed() const = 0;
    virtual void Start() = 0;
    virtual void Stop() = 0;
    virtual void Reset() = 0;
    virtual void Restart() = 0;
};

}  // namespace eacpp