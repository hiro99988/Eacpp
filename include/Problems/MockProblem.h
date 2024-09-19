#pragma once

#include <gmock/gmock.h>

#include <eigen3/Eigen/Core>
#include <vector>

#include "Problems/IProblem.h"

namespace Eacpp {

template <typename T>
class MockProblem : public IProblem<T> {
   public:
    MOCK_CONST_METHOD1_T(ComputeObjectiveSet, Eigen::ArrayXd(const Eigen::ArrayX<T>&));
    MOCK_CONST_METHOD1_T(IsFeasible, bool(const Eigen::ArrayX<T>&));
    MOCK_CONST_METHOD1_T(EvaluateConstraints, std::vector<bool>(const Eigen::ArrayX<T>&));
};

}  // namespace Eacpp