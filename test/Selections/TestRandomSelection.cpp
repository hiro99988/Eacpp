#include <gtest/gtest.h>

#include <eigen3/Eigen/Core>

#include "Selections/RandomSelection.h"

namespace Eacpp::Test {

TEST(RandomSelection, Select) {
    RandomSelection<int> selection;
    Eigen::ArrayXXi population(2, 5);
    population << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
    Eigen::ArrayXXi selected = selection.Select(2, population);
    ASSERT_EQ(population.rows(), selected.rows());
    ASSERT_EQ(2, selected.cols());
}

}  // namespace Eacpp::Test