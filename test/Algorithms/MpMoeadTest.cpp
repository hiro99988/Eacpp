#define _TEST_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <tuple>
#include <vector>

#include "Algorithms/MpMoead.h"
#include "Utils/Utils.h"

using ::testing::Return;

namespace Eacpp {

class MpMoeadTest : public ::testing::Test {
   protected:
    template <typename T>
    std::vector<int> GenerateSolutionIndexes(MpMoead<T>& moead, int totalPopulationSize, int rank, int parallelSize) {
        moead.rank = rank;
        moead.parallelSize = parallelSize;
        return moead.GenerateSolutionIndexes(totalPopulationSize);
    }
    template <typename T>
    std::vector<std::vector<double>> GenerateWeightVectors(MpMoead<T>& moead, int H) {
        return moead.GenerateWeightVectors(H);
    }
    template <typename T>
    std::vector<std::vector<std::pair<double, int>>> CalculateEuclideanDistanceBetweenEachWeightVector(
        MpMoead<T>& moead, int totalPopulationSize, std::vector<double>& weightVectors) {
        return moead.CalculateEuclideanDistanceBetweenEachWeightVector(totalPopulationSize, weightVectors);
    }
    template <typename T>
    std::vector<int> GenerateNeighborhoods(MpMoead<T>& moead, int totalPopulationSize, std::vector<double>& allWeightVectors) {
        return moead.GenerateNeighborhoods(totalPopulationSize, allWeightVectors);
    }
};

}  // namespace Eacpp

namespace Eacpp::Test {

TEST_F(MpMoeadTest, GenerateSolutionIndexes) {
    MpMoead<int> moead = MpMoead<int>(0, 0, 0, 0);
    int totalPopulationSize = 9;
    int parallelSize = 4;

    auto actual = GenerateSolutionIndexes(moead, totalPopulationSize, 0, parallelSize);
    std::vector<int> expected = {0, 1, 2};
    EXPECT_TRUE(actual == expected);

    actual = GenerateSolutionIndexes(moead, totalPopulationSize, 1, parallelSize);
    expected = {3, 4};
    EXPECT_TRUE(actual == expected);

    actual = GenerateSolutionIndexes(moead, totalPopulationSize, 3, parallelSize);
    expected = {7, 8};
    EXPECT_TRUE(actual == expected);
}

TEST_F(MpMoeadTest, GenerateWeightVectors) {
    int objectiveNum = 2;
    int H = 2;
    MpMoead<int> moead = MpMoead<int>(0, 0, objectiveNum, 0);
    auto actual = GenerateWeightVectors(moead, H);

    int expectedSize = 3;
    EXPECT_EQ(actual.size(), expectedSize);

    std::vector<std::vector<double>> expected = {{0.0, 1.0}, {0.5, 0.5}, {1.0, 0.0}};
    for (int i = 0; i < actual.size(); i++) {
        for (int j = 0; j < actual[i].size(); j++) {
            EXPECT_EQ(actual[i].size(), objectiveNum);
            EXPECT_EQ(actual[i][j], expected[i][j]);
        }
    }
}

TEST_F(MpMoeadTest, CalculateEuclideanDistanceBetweenEachWeightVector) {
    int totalPopulationSize = 5;
    int objectiveNum = 2;
    MpMoead<int> moead = MpMoead<int>(0, 0, objectiveNum, 0);
    std::vector<double> weightVectors = {0.0,  1.0,   //
                                         0.25, 0.75,  //
                                         0.5,  0.5,   //
                                         0.75, 0.25,  //
                                         1.0,  0.0};
    std::vector<std::vector<std::pair<double, int>>> expected = {{{0.0, 0}, {0.5, 1}, {1.0, 2}, {1.5, 3}, {2.0, 4}},  //
                                                                 {{0.5, 0}, {0.0, 1}, {0.5, 2}, {1.0, 3}, {1.5, 4}},  //
                                                                 {{1.0, 0}, {0.5, 1}, {0.0, 2}, {0.5, 3}, {1.0, 4}},  //
                                                                 {{1.5, 0}, {1.0, 1}, {0.5, 2}, {0.0, 3}, {0.5, 4}},  //
                                                                 {{2.0, 0}, {1.5, 1}, {1.0, 2}, {0.5, 3}, {0.0, 4}}};
    auto actual = CalculateEuclideanDistanceBetweenEachWeightVector(moead, totalPopulationSize, weightVectors);
    for (int i = 0; i < actual.size(); i++) {
        for (int j = 0; j < actual[i].size(); j++) {
            EXPECT_EQ(actual[i].size(), totalPopulationSize);
            EXPECT_EQ(actual[i][j].first, expected[i][j].first);
            EXPECT_EQ(actual[i][j].second, expected[i][j].second);
        }
    }
}

}  // namespace Eacpp::Test