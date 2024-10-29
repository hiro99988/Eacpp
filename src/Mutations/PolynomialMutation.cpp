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
    if (index < variableBounds.size()) {
        individual.solution(index) += sigma * (variableBounds[index].second - variableBounds[index].first);
    } else {
        individual.solution(index) += sigma * (variableBounds.back().second - variableBounds.back().first);
    }
}

double PolynomialMutation::Sigma() const {
    double random = _rng->Random();
    if (random <= 0.5) {
        return std::pow(2.0 * random, 1.0 / (distributionIndex + 1.0)) - 1.0;
    } else {
        return 1.0 - std::pow(2.0 - 2.0 * random, 1.0 / (distributionIndex + 1.0));
    }
}

}  // namespace Eacpp