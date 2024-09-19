#pragma once

#include <gmock/gmock.h>

#include <eigen3/Eigen/Core>
#include <vector>

#include "Decompositions/IDecomposition.h"

namespace Eacpp {

class MockDecomposition : public IDecomposition {
   public:
    MOCK_CONST_METHOD3(ComputeObjective, double(const Eigen::ArrayXd&, const Eigen::ArrayXd&, const Eigen::ArrayXd&));
};

}  // namespace Eacpp