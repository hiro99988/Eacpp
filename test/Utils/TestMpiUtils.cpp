#include <gtest/gtest.h>

#include <tuple>
#include <vector>

#include "Utils/MpiUtils.h"

namespace Eacpp::Test {

TEST(MpiUtilsTest, CalculateNodeWorkload) {
    int totalPopulationSize = 7;
    int parallelSize = 4;
    std::vector<int> ranks = {0, 2, 3};
    std::vector<int> expected = {2, 2, 1};
    for (int i = 0; i < ranks.size(); i++) {
        int actual = CalculateNodeWorkload(totalPopulationSize, ranks[i], parallelSize);
        EXPECT_EQ(actual, expected[i]);
    }
}

TEST(MpiUtilsTest, CalculateNodeWorkloads) {
    int totalPopulationSize = 7;
    int parallelSize = 4;
    std::vector<int> expected = {2, 2, 2, 1};
    std::vector<int> actual = CalculateNodeWorkloads(totalPopulationSize, parallelSize);
    EXPECT_EQ(actual, expected);
}

TEST(MpiUtilsTest, CalculateNodeStartIndex) {
    int totalPopulationSize = 7;
    int parallelSize = 4;
    std::vector<int> ranks = {0, 2, 3};
    std::vector<int> expected = {0, 4, 6};
    for (int i = 0; i < ranks.size(); i++) {
        int actual = CalculateNodeStartIndex(totalPopulationSize, ranks[i], parallelSize);
        EXPECT_EQ(actual, expected[i]);
    }
}

TEST(MpiUtilsTest, GenerateDataCountsAndDisplacements) {
    std::vector<int> nodeWorkloads = {2, 2, 2, 1};
    int dataSize = 3;
    int parallelSize = 4;
    std::vector<int> expectedDataCounts = {6, 6, 6, 3};
    std::vector<int> expectedDisplacements = {0, 6, 12, 18};
    auto actual = GenerateDataCountsAndDisplacements(nodeWorkloads, dataSize, parallelSize);
    EXPECT_EQ(actual.first, expectedDataCounts);
    EXPECT_EQ(actual.second, expectedDisplacements);
}

}  // namespace Eacpp::Test