#include "Mutations/PolynomialMutation.h"

#include <cmath>
#include <eigen3/Eigen/Core>

namespace Eacpp {

void PolynomialMutation::Mutate(Eigen::ArrayXd& individual) const {
    for (int i = 0; i < individual.size(); ++i) {
        if (_rng->Random() > mutationRate) {
            continue;
        }

        PerformMutation(i, individual, Sigma());
    }
}

void PolynomialMutation::PerformMutation(int index, Eigen::ArrayXd& individual, double sigma) const {
    if (variableBounds.size() == 1) {
        individual(index) += sigma * (variableBounds[0][1] - variableBounds[0][0]);
    } else if (index < _lastBoundIndex) {
        individual(index) += sigma * (variableBounds[index][1] - variableBounds[index][0]);
    } else {
        individual(index) += sigma * (variableBounds[_lastBoundIndex][1] - variableBounds[_lastBoundIndex][0]);
    }
}

double PolynomialMutation::Sigma() const {
    if (_rng->Random() <= 0.5) {
        return std::pow(2.0 * _rng->Random(), 1.0 / (distributionIndex + 1.0)) - 1.0;
    } else {
        return 1.0 - std::pow(2.0 - 2.0 * _rng->Random(), 1.0 / (distributionIndex + 1.0));
    }
}

}  // namespace Eacpp