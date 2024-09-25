#include "Decompositions/Tchebycheff.h"

#include <eigen3/Eigen/Dense>

namespace Eacpp {

double Tchebycheff::ComputeObjective(const Eigen::ArrayXd& weight, const Eigen::ArrayXd& objectiveSet) const {
    Eigen::ArrayXd absDiff = (objectiveSet - _idealPoint).abs();
    Eigen::ArrayXd weightedDiff = weight * absDiff;
    return weightedDiff.maxCoeff();
}

}  // namespace Eacpp