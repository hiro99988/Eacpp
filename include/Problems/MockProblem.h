#pragma once

#include <gmock/gmock.h>

#include <eigen3/Eigen/Core>
#include <vector>

#include "Problems/IProblem.h"

namespace Eacpp {

template <typename T>
class MockProblem : public IProblem<T> {
   public:
    MOCK_CONST_METHOD1(ComputeObjectiveSet, Eigen::ArrayXd(const Eigen::ArrayX<T>&));
    MOCK_CONST_METHOD1(IsFeasible, bool(const Eigen::ArrayX<T>&));
    MOCK_CONST_METHOD1(EvaluateConstraints, std::vector<bool>(const Eigen::ArrayX<T>&));
};

}  // namespace Eacpp