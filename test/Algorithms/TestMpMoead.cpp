#define _TEST_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "Algorithms/MpMoead.h"
#include "Utils/Utils.h"

using ::testing::Return;

namespace Eacpp {

class MpMpeadTest : public ::testing::Test {
   protected:
    template <typename T>
    std::vector<double> GenerateAllWeightVectors(MpMoead<T>& moead, int H) {
        return moead.GenerateAllWeightVectors(H);
    }
    template <typename T>
    std::vector<int> GenerateAllNeighborhoods(MpMoead<T>& moead, int totalPopulationSize,
                                              std::vector<double>& allWeightVectors) {
        return moead.GenerateNeighborhoods(totalPopulationSize, allWeightVectors);
    }
};

}  // namespace Eacpp

namespace Eacpp::Test {

TEST_F(MpMpeadTest, GenerateWeightVectors) {
    int objectiveNum = 3;
    int H = 3;
    MpMoead<int> moead = MpMoead<int>(0, 0, objectiveNum, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    std::vector<double> actual = GenerateAllWeightVectors(moead, H);

    int n = Combination(H + objectiveNum - 1, objectiveNum - 1);
    int expectedSize = n * objectiveNum;
    EXPECT_EQ(actual.size(), expectedSize);

    // 各重みベクトルの各要素の合計がHになっているか確認
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < objectiveNum; j++) {
            sum += actual[i * objectiveNum + j];
        }
        EXPECT_EQ(sum, H);
    }
}

TEST_F(MpMpeadTest, GenerateNeighborhoods) {
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