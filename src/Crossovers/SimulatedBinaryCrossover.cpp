#include "Crossovers/SimulatedBinaryCrossover.h"

#include <cmath>
#include <eigen3/Eigen/Core>
#include <memory>
#include <tuple>
#include <vector>

#include "Individual/Individual.h"

namespace Eacpp {

Individuald SimulatedBinaryCrossover::performCrossover(const std::vector<Individuald>& parents) const {
    const Eigen::VectorXd& parent1 = parents[0].solution;
    const Eigen::VectorXd& parent2 = parents[1].solution;
    Eigen::VectorXd child = parent1;

    for (int i = 0; i < child.size(); i++) {
        if (_rng->Random() > crossoverRate) {
            continue;
        }

        std::pair<double, double> bound;
        if (i < variableBounds.size()) {
            bound = variableBounds[i];
        } else {
            bound = variableBounds.back();
        }

        double beta;
        bool isOne = _rng->Random() < 0.5;
        if (isOne) {
            beta = Beta1(parent1(i), parent2(i), bound.first);
        } else {
            beta = Beta2(parent1(i), parent2(i), bound.second);
        }

        double alpha = Alpha(beta);
        double betaq = Betaq(alpha);
        if (isOne) {
            child(i) = 0.5 * ((1.0 + betaq) * parent1(i) + (1.0 - betaq) * parent2(i));
        } else {
            child(i) = 0.5 * ((1.0 - betaq) * parent1(i) + (1.0 + betaq) * parent2(i));
        }
    }

    return Individuald(child);
}

double SimulatedBinaryCrossover::Betaq(double alpha) const {
    double betaq;
    double random = _rng->Random();
    if (random <= 1.0 / alpha) {
        betaq = std::pow(random * alpha, 1.0 / (distributionIndex + 1.0));
    } else {
        betaq = std::pow(1.0 / (2.0 - random * alpha), 1.0 / (distributionIndex + 1.0));
    }

    return betaq;
}

double SimulatedBinaryCrossover::Alpha(double beta) const {
    return 2.0 - std::pow(beta, -(distributionIndex + 1.0));
}

double SimulatedBinaryCrossover::Beta1(double x1, double x2, double lowerBound) const {
    return 1.0 + 2.0 * (x1 - lowerBound) / (x2 - x1);
}

double SimulatedBinaryCrossover::Beta2(double x1, double x2, double upperBound) const {
    return 1.0 + 2.0 * (upperBound - x2) / (x2 - x1);
}

}  // namespace Eacpp