#include <gtest/gtest.h>

#include <eigen3/Eigen/Core>

#include "Selections/RandomSelection.h"

namespace Eacpp::Test {

TEST(RandomSelection, Select) {
    RandomSelection selection;
    std::vector<int> population = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> selected = selection.Select(2, population);
    ASSERT_EQ(2, selected.size());
}

}  // namespace Eacpp::Test