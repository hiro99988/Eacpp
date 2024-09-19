#pragma once

#include <eigen3/Eigen/Core>
#include <vector>

namespace Eacpp {

template <typename T>
struct IProblem {
    virtual ~IProblem() {}

    virtual Eigen::ArrayXd ComputeObjectiveSet(const Eigen::ArrayX<T>& solution) const = 0;
    virtual bool IsFeasible(const Eigen::ArrayX<T>& solution) const = 0;
    virtual std::vector<bool> EvaluateConstraints(const Eigen::ArrayX<T>& solution) const = 0;
};

}  // namespace Eacpp
