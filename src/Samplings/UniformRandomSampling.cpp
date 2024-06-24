#include "Samplings/UniformRandomSampling.h"

#include <eigen3/Eigen/Dense>
#include <iostream>

namespace Eacpp {

Eigen::ArrayXXd UniformRandomSampling::Sample(int sampleNum, int variableNum) const {
    return _rng.Uniform(min, max, {sampleNum, variableNum});
}

}  // namespace Eacpp