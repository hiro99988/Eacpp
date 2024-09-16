#pragma once

#include <gmock/gmock.h>

#include <eigen3/Eigen/Core>
#include <vector>

#include "Samplings/ISampling.h"

namespace Eacpp {

template <typename T>
class MockSampling : public ISampling<T> {
   public:
    MOCK_CONST_METHOD2(Sample, std::vector<Eigen::ArrayX<T>>(const int, const int));
};

}  // namespace Eacpp