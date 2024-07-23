#include "Mutations/PolynomialMutation.h"

#include <cmath>
#include <eigen3/Eigen/Core>

namespace Eacpp {

void PolynomialMutation::Mutate(Eigen::ArrayXd& individual) const {
    for (int i = 0; i < individual.size(); ++i) {
        if (_rng->Random() >= mutationRate) {
            continue;
        }
        if (_length == 1) {
            individual[i] += Sigma() * (variableBounds[0][1] - variableBounds[0][0]);
        } else {
            if (i < _lastBoundIndex) {
                individual(i) += Sigma() * (variableBounds[i][1] - variableBounds[i][0]);
            } else {
                individual(i) += Sigma() * (variableBounds[_lastBoundIndex][1] - variableBounds[_lastBoundIndex][0]);
            }
        }
    }
}

double PolynomialMutation::Sigma() const {
    if (_rng->Random() < 0.5) {
        return std::pow(2.0 * _rng->Random(), 1.0 / (distributionIndex + 1.0)) - 1.0;
    } else {
        return 1.0 - std::pow(2.0 - 2.0 * _rng->Random(), 1.0 / (distributionIndex + 1.0));
    }
}

}  // namespace Eacpp