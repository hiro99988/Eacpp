#include "Samplings/UniformRandomSampling.h"

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>

namespace Eacpp {

std::vector<Individuald> UniformRandomSampling::Sample(const int sampleNum, const int variableNum) const {
    std::vector<Eigen::ArrayXd> solutions = _rng->Uniform(min, max, {sampleNum, variableNum});
    std::vector<Individuald> individuals;
    individuals.reserve(sampleNum);
    for (const auto& solution : solutions) {
        individuals.emplace_back(solution);
    }
    return individuals;
}

}  // namespace Eacpp