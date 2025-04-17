#pragma once

#include <Eigen/Core>

namespace Eacpp {

template <typename DecisionVariableType>
struct IMoead {
    virtual ~IMoead() {}

    virtual int CurrentGeneration() const = 0;
    virtual void Initialize() = 0;
    virtual void Update() = 0;
    virtual void Run() = 0;
    virtual bool IsEnd() const = 0;
    virtual std::vector<Eigen::ArrayXd> GetObjectivesList() const = 0;
    virtual std::vector<Eigen::ArrayX<DecisionVariableType>> GetSolutionList()
        const = 0;
};

}  // namespace Eacpp
