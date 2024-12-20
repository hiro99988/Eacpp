#pragma once

#include <eigen3/Eigen/Core>

#include "Individual.h"

namespace Eacpp {

struct IDecomposition {
    virtual ~IDecomposition() {}

    virtual const Eigen::ArrayXd& IdealPoint() const = 0;
    virtual double ComputeObjective(
        const Eigen::ArrayXd& weight,
        const Eigen::ArrayXd& objectiveSet) const = 0;
    virtual void InitializeIdealPoint(int objectivesNum) = 0;
    virtual void UpdateIdealPoint(const Eigen::ArrayXd& objectiveSet) = 0;
};

}  // namespace Eacpp
