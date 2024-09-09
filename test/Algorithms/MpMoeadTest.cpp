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

TEST_F(MpMoeadTest, GenerateNeighborhoods) {
    int totalPopulationSize = 4;
    int objectiveNum = 2;
    int neighborNum = 3;
    std::vector<double> allWeightVectors = {0.0, 1.0, 0.2, 0.8, 0.8, 0.2, 1.0, 0.0};
    MpMoead<int> moead = MpMoead<int>(0, 0, objectiveNum, neighborNum, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    std::vector<int> actual = GenerateAllNeighborhoods(moead, totalPopulationSize, allWeightVectors);

    int expectedSize = totalPopulationSize * neighborNum;
    EXPECT_EQ(actual.size(), expectedSize);

    // 各重みベクトルの近傍が正しく設定されているか確認
    std::vector<int> expected = {0, 1, 2, 1, 0, 2, 2, 3, 1, 3, 2, 1};
    for (int i = 0; i < expectedSize; i++) {
        EXPECT_EQ(actual[i], expected[i]);
    }
}

}  // namespace Eacpp::Test