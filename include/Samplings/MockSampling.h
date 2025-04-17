#pragma once

#include <gmock/gmock.h>

#include <Eigen/Core>
#include <vector>

#include "Individual.h"
#include "Samplings/ISampling.h"

namespace Eacpp {

template <typename T>
class MockSampling : public ISampling<T> {
   public:
    MOCK_CONST_METHOD2_T(Sample,
                         std::vector<Individual<T>>(const int, const int));
};

}  // namespace Eacpp