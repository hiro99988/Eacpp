#include "Samplings/UniformRandomSampling.h"

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>

namespace Eacpp {

std::vector<Eigen::ArrayXd> UniformRandomSampling::Sample(const int sampleNum, const int variableNum) const {
    return _rng->Uniform(min, max, {variableNum, sampleNum});
}

}  // namespace Eacpp