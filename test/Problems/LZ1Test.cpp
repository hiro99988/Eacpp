#include <gtest/gtest.h>

#include <cmath>

#include "Problems/LZ1.h"

namespace Eacpp::Test {

TEST(LZ1Test, ComputeObjectiveSet) {
    LZ1 lz1(3);
    Eigen::ArrayXd solution(3);
    solution << 0.2, 0.5, 0.7;
    Eigen::ArrayXd objectiveSet = lz1.ComputeObjectiveSet(solution);
    double expectedOb1 = 0.2 + 2.0 * std::pow(0.66, 2);
    double expectedOb2 = 1.0 - std::sqrt(0.2) + 2.0 * std::pow(0.5 - std::pow(0.2, 0.5), 2);
    ASSERT_DOUBLE_EQ(expectedOb1, objectiveSet(0));
    ASSERT_DOUBLE_EQ(expectedOb2, objectiveSet(1));
}

}  // namespace Eacpp::Test