#pragma once

#include <eigen3/Eigen/Core>

#include "Decompositions/DecompositionBase.h"

namespace Eacpp {

class Tchebycheff : public DecompositionBase {
   public:
    Tchebycheff() {}

    double ComputeObjective(const Eigen::ArrayXd& weight, const Eigen::ArrayXd& objectiveSet) const override;
};

}  // namespace Eacpp
