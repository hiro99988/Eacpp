#pragma once

#include <eigen3/Eigen/Core>
#include <vector>

#include "Individual/Individual.h"

namespace Eacpp {

template <typename T>
struct IProblem {
    virtual ~IProblem() {}

    virtual int DecisionVariablesNum() const = 0;
    virtual int ObjectivesNum() const = 0;
    virtual void ComputeObjectiveSet(Individual<T>& individual) const = 0;
    virtual bool IsFeasible(const Individual<T>& individual) const = 0;
    virtual std::vector<bool> EvaluateConstraints(const Individual<T>& individual) const = 0;
};

}  // namespace Eacpp
