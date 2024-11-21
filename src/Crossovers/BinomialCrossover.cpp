#include "Crossovers/BinomialCrossover.h"

#include <eigen3/Eigen/Core>
#include <vector>

#include "Individual.h"

namespace Eacpp {

Individuald BinomialCrossover::performCrossover(const std::vector<Individuald>& parents) const {
    Individuald child(parents[0].solution);
    int jr = _rng->Integer(child.solution.size() - 1);
    for (int i = 0; i < child.solution.size(); i++) {
        if (_rng->Random() <= crossoverRate || i == jr) {
            child.solution(i) += scalingFactor * (parents[1].solution(i) - parents[2].solution(i));
        }
    }
    return child;
}

}  // namespace Eacpp