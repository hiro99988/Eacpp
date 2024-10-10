#pragma once

#include <eigen3/Eigen/Core>

namespace Eacpp {

template <typename DecisionVariableType>
struct IMoead {
    virtual void Initialize() = 0;
    virtual void Update() = 0;
    virtual void Run() = 0;
    virtual double GetExecutionTime() = 0;
    virtual std::vector<Eigen::ArrayXd> GetObjectivesList() = 0;
    virtual std::vector<Eigen::ArrayX<DecisionVariableType>> GetSolutionList() = 0;
};

}  // namespace Eacpp
