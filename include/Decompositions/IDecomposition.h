#pragma once

#include <eigen3/Eigen/Core>

#include "Individual/Individual.h"

namespace Eacpp {

struct IDecomposition {
    virtual ~IDecomposition() {}

    virtual Eigen::ArrayXd& IdealPoint() = 0;
    virtual double ComputeObjective(const Eigen::ArrayXd& weight, const Eigen::ArrayXd& objectiveSet) const = 0;
    virtual void UpdateIdealPoint(const Eigen::ArrayXd& objectiveSet) = 0;
};

}  // namespace Eacpp
