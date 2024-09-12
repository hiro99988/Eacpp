#define _TEST_

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include <eigen3/Eigen/Core>

#include "Algorithms/MpMoead.h"

// -n 4 で実行すること。

namespace Eacpp {

class MpMoeadMpiTest : public ::testing::Test {
   protected:
    static void SetUpTestSuite() { MPI_Init(nullptr, nullptr); }

    static void TearDownTestSuite() { MPI_Finalize(); }

    template <typename T>
    int GetRank(MpMoead<T>& moead) {
        return moead.rank;
    }
    template <typename T>
    std::vector<int> GetSolutionIndexes(MpMoead<T>& moead) {
        return moead.solutionIndexes;
    }
    template <typename T>
    std::vector<int> GetExternalSolutionIndexes(MpMoead<T>& moead) {
        return moead.externalSolutionIndexes;
    }
    // TODO: Individualもテストする
    // template <typename T>
    // std::unordered_map<int, MpMoead<T>::Individual> GetIndividuals(MpMoead<T>& moead) {
    //     return moead.individuals;
    // }
    template <typename T>
    std::unordered_map<int, Eigen::ArrayXd> GetWeightVectors(MpMoead<T>& moead) {
        return moead.weightVectors;
    }
};

}  // namespace Eacpp

namespace Eacpp::Test {

TEST_F(MpMoeadMpiTest, InitializeIsland) {
    int totalPopulationSize = 9;
    int H = 8;
    MpMoead<int> moead = MpMoead<int>(totalPopulationSize, 0, 0, 2, 3);
    moead.InitializeMpi(0, nullptr);
    moead.InitializeIsland(H);

    std::vector<int> expectedSolutionIndexes;
    std::vector<int> expectedExternalSolutionIndexes;
    // std::unordered_map<int, Individual> individuals;
    std::unordered_map<int, Eigen::ArrayXd> expectedWeightVectors;

    if (GetRank(moead) == 0) {
        expectedSolutionIndexes = {0, 1, 2};
        expectedExternalSolutionIndexes = {3};

        EXPECT_TRUE(GetSolutionIndexes(moead) == expectedSolutionIndexes);
        EXPECT_TRUE(GetExternalSolutionIndexes(moead) == expectedExternalSolutionIndexes);
        EXPECT_EQ(GetWeightVectors(moead).size(), expectedSolutionIndexes.size() + expectedExternalSolutionIndexes.size());
        for (int i = 0; i < GetWeightVectors(moead).size(); i++) {
            Eigen::ArrayXd wv(2);
            wv << (double)i / H, 1.0 - (double)i / H;
            EXPECT_TRUE((GetWeightVectors(moead)[i] == wv).all());
        }
    } else if (GetRank(moead) == 1) {
        expectedSolutionIndexes = {3, 4};
        expectedExternalSolutionIndexes = {2, 5};

        EXPECT_TRUE(GetSolutionIndexes(moead) == expectedSolutionIndexes);
        EXPECT_TRUE(GetExternalSolutionIndexes(moead) == expectedExternalSolutionIndexes);
        EXPECT_EQ(GetWeightVectors(moead).size(), expectedSolutionIndexes.size() + expectedExternalSolutionIndexes.size());
        for (int i = 2; i < 2 + GetWeightVectors(moead).size(); i++) {
            Eigen::ArrayXd wv(2);
            wv << (double)i / H, 1.0 - (double)i / H;
            EXPECT_TRUE((GetWeightVectors(moead)[i] == wv).all());
        }
    } else if (GetRank(moead) == 2) {
        expectedSolutionIndexes = {5, 6};
        expectedExternalSolutionIndexes = {4, 7};

        EXPECT_TRUE(GetSolutionIndexes(moead) == expectedSolutionIndexes);
        EXPECT_TRUE(GetExternalSolutionIndexes(moead) == expectedExternalSolutionIndexes);
        EXPECT_EQ(GetWeightVectors(moead).size(), expectedSolutionIndexes.size() + expectedExternalSolutionIndexes.size());
        for (int i = 4; i < 4 + GetWeightVectors(moead).size(); i++) {
            Eigen::ArrayXd wv(2);
            wv << (double)i / H, 1.0 - (double)i / H;
            EXPECT_TRUE((GetWeightVectors(moead)[i] == wv).all());
        }
    } else if (GetRank(moead) == 3) {
        expectedSolutionIndexes = {7, 8};
        expectedExternalSolutionIndexes = {6};

        EXPECT_TRUE(GetSolutionIndexes(moead) == expectedSolutionIndexes);
        EXPECT_TRUE(GetExternalSolutionIndexes(moead) == expectedExternalSolutionIndexes);
        EXPECT_EQ(GetWeightVectors(moead).size(), expectedSolutionIndexes.size() + expectedExternalSolutionIndexes.size());
        for (int i = 6; i < 6 + GetWeightVectors(moead).size(); i++) {
            Eigen::ArrayXd wv(2);
            wv << (double)i / H, 1.0 - (double)i / H;
            EXPECT_TRUE((GetWeightVectors(moead)[i] == wv).all());
        }
    }
}

}  // namespace Eacpp::Test