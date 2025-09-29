#include "Decompositions/Tchebycheff.h"

#include <Eigen/Dense>

namespace eacpp {

double Tchebycheff::ComputeObjective(const Eigen::ArrayXd& weight,
                                     const Eigen::ArrayXd& objectiveSet) const {
    Eigen::ArrayXd absDiff = (objectiveSet - _idealPoint).abs();
    Eigen::ArrayXd weightedDiff = weight * absDiff;
    return weightedDiff.maxCoeff();
}

}  // namespace eacpp