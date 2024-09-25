#include "Mutations/PolynomialMutation.h"

#include <cmath>
#include <eigen3/Eigen/Core>

#include "Individual/Individual.h"

namespace Eacpp {

void PolynomialMutation::Mutate(Individuald& individual) const {
    for (int i = 0; i < individual.solution.size(); ++i) {
        if (_rng->Random() > mutationRate) {
            continue;
        }

        PerformMutation(i, individual, Sigma());
    }
}

void PolynomialMutation::PerformMutation(int index, Individuald& individual, double sigma) const {
    if (variableBounds.size() == 1) {
        individual.solution(index) += sigma * (variableBounds[0].second - variableBounds[0].first);
    } else if (index < _lastBoundIndex) {
        individual.solution(index) += sigma * (variableBounds[index].second - variableBounds[index].first);
    } else {
        individual.solution(index) += sigma * (variableBounds[_lastBoundIndex].second - variableBounds[_lastBoundIndex].first);
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