#include "Crossovers/BinomialCrossover.h"

#include <eigen3/Eigen/Core>
#include <vector>

namespace Eacpp {

Eigen::ArrayXd BinomialCrossover::performCrossover(const std::vector<Eigen::ArrayXd>& parents) const {
    Eigen::ArrayXd child = parents[0];
    int jr = _rng->Integer(child.size() - 1);
    for (int i = 0; i < child.size(); i++) {
        if (_rng->Random() <= crossoverRate || i == jr) {
            child(i) += scalingFactor * (parents[1](i) - parents[2](i));
        }
    }
    return child;
}

}  // namespace Eacpp