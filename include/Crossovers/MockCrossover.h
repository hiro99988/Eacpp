#pragma once

#include <gmock/gmock.h>

#include <eigen3/Eigen/Core>
#include <vector>

#include "Crossovers/ICrossover.h"

namespace Eacpp {

template <typename T>
class MockCrossover : public ICrossover<T> {
   public:
    MOCK_CONST_METHOD0_T(GetParentNum, int());
    MOCK_CONST_METHOD1_T(Cross, Eigen::ArrayX<T>(const std::vector<Eigen::ArrayX<T>>&));
};

}  // namespace Eacpp