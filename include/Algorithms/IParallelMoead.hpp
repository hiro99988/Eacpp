#pragma once

#include "Algorithms/IMoead.h"

namespace eacpp {

template <typename DecisionVariableType>
struct IParallelMoead : public IMoead<DecisionVariableType> {
    virtual ~IParallelMoead() {}

    virtual double GetInitializationTime() const = 0;
    virtual double GetExecutionTime() const = 0;
    virtual double GetCommunicationTime() const = 0;
    virtual std::vector<std::vector<int>> GetDataTraffics() const = 0;
};

}  // namespace eacpp