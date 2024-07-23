#include "Samplings/UniformRandomSampling.h"

#include <eigen3/Eigen/Dense>
#include <iostream>

namespace Eacpp {

Eigen::ArrayXXd UniformRandomSampling::Sample(int sampleNum, int variableNum) {
    return _rng->Uniform(min, max, {variableNum, sampleNum});
}

}  // namespace Eacpp