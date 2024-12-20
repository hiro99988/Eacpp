#include "Mutations/PolynomialMutation.h"

#include <cmath>
#include <eigen3/Eigen/Core>

#include "Individual.h"

namespace Eacpp {

void PolynomialMutation::Mutate(Individuald& individual) const {
    for (int i = 0; i < individual.solution.size(); ++i) {
        if (_rng->Random() >= mutationRate) {
            continue;
        }

        double yl, yu;
        if (i < variableBounds.size()) {
            yl = variableBounds[i].first;
            yu = variableBounds[i].second;
        } else {
            yl = variableBounds.back().first;
            yu = variableBounds.back().second;
        }

        individual.solution(i) =
            PerformMutation(individual.solution(i), yl, yu);
    }
}

double PolynomialMutation::PerformMutation(double y, double lower,
                                           double upper) const {
    double delta1 = (y - lower) / (upper - lower);
    double delta2 = (upper - y) / (upper - lower);
    double mut_pow = 1.0 / (distributionIndex + 1.0);
    double rand = _rng->Random();

    double deltaq;
    if (rand <= 0.5) {
        double xy = 1.0 - delta1;
        double val = 2.0 * rand +
                     (1.0 - 2.0 * rand) * std::pow(xy, distributionIndex + 1.0);
        deltaq = std::pow(val, mut_pow) - 1.0;
    } else {
        double xy = 1.0 - delta2;
        double val = 2.0 * (1.0 - rand) +
                     2.0 * (rand - 0.5) * std::pow(xy, distributionIndex + 1.0);
        deltaq = 1.0 - std::pow(val, mut_pow);
    }

    y = y + deltaq * (upper - lower);

    y = std::min(std::max(y, lower), upper);

    return y;
}

}  // namespace Eacpp