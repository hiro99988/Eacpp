#pragma once

#include <gmock/gmock.h>

#include <eigen3/Eigen/Core>

#include "Decompositions/IDecomposition.h"

namespace Eacpp {

class MockDecomposition : public IDecomposition {
   public:
    MOCK_METHOD0(IdealPoint, Eigen::ArrayXd&());
    MOCK_CONST_METHOD2(ComputeObjective, double(const Eigen::ArrayXd&, const Eigen::ArrayXd&));
    MOCK_METHOD1(UpdateIdealPoint, void(const Eigen::ArrayXd&));
};

}  // namespace Eacpp