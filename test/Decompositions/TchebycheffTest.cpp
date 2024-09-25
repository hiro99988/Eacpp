#include <gtest/gtest.h>

#include <eigen3/Eigen/Core>

#include "Decompositions/Tchebycheff.h"

namespace Eacpp::Test {

TEST(TchebycheffTest, ComputeObjective) {
    Eacpp::Tchebycheff tchebycheff(3);
    Eigen::ArrayXd weight(3);
    Eigen::ArrayXd objectiveSet(3);
    weight << 0.5, 0.4, 0.1;
    objectiveSet << 2, 4, 6;
    tchebycheff.IdealPoint() << 1, 2, 3;

    double actual = tchebycheff.ComputeObjective(weight, objectiveSet);
    double expected = 0.8;
    ASSERT_DOUBLE_EQ(expected, actual);
}

}  // namespace Eacpp::Test