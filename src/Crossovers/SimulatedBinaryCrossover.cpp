#include "Crossovers/SimulatedBinaryCrossover.h"

#include <cmath>
#include <eigen3/Eigen/Core>

namespace Eacpp {

Individuald SimulatedBinaryCrossover::performCrossover(const std::vector<Individuald>& parents) const {
    const Eigen::VectorXd& parent1 = parents[0].solution;
    const Eigen::VectorXd& parent2 = parents[1].solution;
    Eigen::VectorXd child = parent1;

    for (int i = 0; i < child.size(); i++) {
        if (_rng->Random() > crossoverRate) {
            continue;
        }

        double beta = Beta();
        child(i) = 0.5 * ((1.0 + beta) * parent1(i) + (1.0 - beta) * parent2(i));
    }

    return Individuald(child);
}

double SimulatedBinaryCrossover::Beta() const {
    double beta;
    double random = _rng->Random();
    if (random <= 0.5) {
        beta = std::pow(2.0 * random, 1.0 / (distributionIndex + 1.0));
    } else {
        beta = std::pow(1.0 / (2.0 * (1.0 - random)), 1.0 / (distributionIndex + 1.0));
    }
    return beta;
}

}  // namespace Eacpp