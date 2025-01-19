#pragma once

#include "Algorithms/IMoead.h"

namespace Eacpp {

template <typename DecisionVariableType>
struct IParallelMoead : public IMoead<DecisionVariableType> {
    virtual ~IParallelMoead() {}

    virtual double GetElapsedTime() const = 0;
    virtual std::vector<std::vector<int>> GetSendDataTraffics() const = 0;
    virtual std::vector<std::vector<int>> GetReceiveDataTraffics() const = 0;
};

}  // namespace Eacpp