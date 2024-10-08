#pragma once

#include <gmock/gmock.h>

#include <eigen3/Eigen/Core>
#include <vector>

#include "Individual/Individual.h"
#include "Problems/IProblem.h"

namespace Eacpp {

template <typename T>
class MockProblem : public IProblem<T> {
   public:
    MOCK_CONST_METHOD1_T(ComputeObjectiveSet, void(Individual<T>&));
    MOCK_CONST_METHOD1_T(IsFeasible, bool(const Individual<T>&));
    MOCK_CONST_METHOD1_T(EvaluateConstraints, std::vector<bool>(const Individual<T>&));
};

}  // namespace Eacpp