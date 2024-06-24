#ifndef IProblem_h
#define IProblem_h

#include <eigen3/Eigen/Core>
#include <vector>

#include "Utils/TemplateType.h"

namespace Eacpp {

struct IProblem {
    virtual Eigen::ArrayXd ComputeObjectiveSet(const Eigen::ArrayXd& solution) const = 0;
    virtual bool IsFeasible(const Eigen::ArrayXd& solution) const = 0;
    virtual std::vector<bool> EvaluateConstraints(const Eigen::ArrayXd& solution) const = 0;
};

}  // namespace Eacpp

#endif