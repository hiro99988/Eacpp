#ifndef IProblem_h
#define IProblem_h

#include <eigen3/Eigen/Core>
#include <vector>

#include "Utils/TemplateType.h"

namespace Eacpp {

template <Number T>
struct IProblem {
    virtual ~IProblem() {}

    virtual Eigen::ArrayXd ComputeObjectiveSet(const Eigen::ArrayX<T>& solution) const = 0;
    virtual bool IsFeasible(const Eigen::ArrayX<T>& solution) const = 0;
    virtual std::vector<bool> EvaluateConstraints(const Eigen::ArrayX<T>& solution) const = 0;
};

}  // namespace Eacpp

#endif