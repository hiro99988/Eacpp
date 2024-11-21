#include "Crossovers/SimulatedBinaryCrossover.h"

#include <cmath>
#include <eigen3/Eigen/Core>
#include <memory>
#include <tuple>
#include <vector>

#include "Individual.h"
#include "Utils/Utils.h"

namespace Eacpp {

Individuald SimulatedBinaryCrossover::performCrossover(const std::vector<Individuald>& parents) const {
    constexpr double epsilon = 1e-14;

    const Eigen::ArrayXd& parent1 = parents[0].solution;
    const Eigen::ArrayXd& parent2 = parents[1].solution;
    Eigen::ArrayXd child(parent1.size());

    for (int i = 0; i < child.size(); i++) {
        double x1 = parent1(i);
        double x2 = parent2(i);
        if (_rng->Random() > crossoverRate || x1 == x2 || std::abs(x1 - x2) <= epsilon) {
            if (_rng->Random() < 0.5) {
                child(i) = x1;
            } else {
                child(i) = x2;
            }
            continue;
        }

        swapIfMaxLessThanMin(x1, x2);

        std::pair<double, double> bound;
        if (i < variableBounds.size()) {
            bound = variableBounds[i];
        } else {
            bound = variableBounds.back();
        }

        double beta;
        bool isOne = _rng->Random() < 0.5;
        if (isOne) {
            beta = Beta1(x1, x2, bound.first);
        } else {
            beta = Beta2(x1, x2, bound.second);
        }

        double alpha = Alpha(beta);
        double betaq = Betaq(alpha);
        if (isOne) {
            child(i) = 0.5 * ((1.0 + betaq) * x1 + (1.0 - betaq) * x2);
        } else {
            child(i) = 0.5 * ((1.0 - betaq) * x1 + (1.0 + betaq) * x2);
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