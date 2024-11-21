#pragma once

#include <eigen3/Eigen/Core>
#include <vector>

#include "Individual.h"

namespace Eacpp {

template <typename T>
struct ISampling {
    virtual ~ISampling() {}

    virtual std::vector<Individual<T>> Sample(const int sampleNum, const int variableNum) const = 0;
};

}  // namespace Eacpp
