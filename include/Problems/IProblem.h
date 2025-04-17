#pragma once

#include <Eigen/Core>
#include <tuple>
#include <vector>

#include "Individual.h"

namespace Eacpp {

template <typename T>
struct IProblem {
    virtual ~IProblem() {}

    virtual int DecisionVariablesNum() const = 0;
    virtual int ObjectivesNum() const = 0;
    virtual const std::vector<std::pair<T, T>>& VariableBounds() const = 0;
    virtual void ComputeObjectiveSet(Individual<T>& individual) const = 0;
    virtual bool IsFeasible(const Individual<T>& individual) const = 0;
    virtual std::vector<bool> EvaluateConstraints(
        const Individual<T>& individual) const = 0;
};

}  // namespace Eacpp
