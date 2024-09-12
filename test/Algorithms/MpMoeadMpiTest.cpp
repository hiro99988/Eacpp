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
    template <typename T>
    std::unordered_map<int, typename MpMoead<T>::Individual> GetIndividuals(MpMoead<T>& moead) {
        return moead.individuals;
    }
    template <typename T>
    std::unordered_map<int, Eigen::ArrayXd> GetWeightVectors(MpMoead<T>& moead) {
        return moead.weightVectors;
    }
    template <typename T>
    std::pair<std::vector<int>, std::vector<double>> ScatterExternalNeighborhood(MpMoead<T>& moead,
                                                                                 std::vector<int>& neighborhoodIndexes,
                                                                                 std::vector<int>& neighborhoodSizes,
                                                                                 std::vector<double>& weightVectors) {
        return moead.ScatterExternalNeighborhood(neighborhoodIndexes, neighborhoodSizes, weightVectors);
    }
};

}  // namespace Eacpp

namespace Eacpp::Test {

TEST_F(MpMoeadMpiTest, InitializeIsland) {
    int totalPopulationSize = 9;
    int H = 8;
    int neighborNum = 3;
    MpMoead<int> moead = MpMoead<int>(totalPopulationSize, 0, 0, 2, neighborNum);
    moead.InitializeMpi(0, nullptr);
    moead.InitializeIsland(H);

    std::vector<int> expectedSolutionIndexes;
    std::vector<int> expectedExternalSolutionIndexes;
    std::vector<int> expectedNeighborhood(neighborNum);

    if (GetRank(moead) == 0) {
        expectedSolutionIndexes = {0, 1, 2};
        expectedExternalSolutionIndexes = {3};
        EXPECT_TRUE(GetSolutionIndexes(moead) == expectedSolutionIndexes);
        EXPECT_TRUE(GetExternalSolutionIndexes(moead) == expectedExternalSolutionIndexes);
    } else if (GetRank(moead) == 1) {
        expectedSolutionIndexes = {3, 4};
        expectedExternalSolutionIndexes = {2, 5};
        EXPECT_TRUE(GetSolutionIndexes(moead) == expectedSolutionIndexes);
        EXPECT_TRUE(GetExternalSolutionIndexes(moead) == expectedExternalSolutionIndexes);
    } else if (GetRank(moead) == 2) {
        expectedSolutionIndexes = {5, 6};
        expectedExternalSolutionIndexes = {4, 7};
        EXPECT_TRUE(GetSolutionIndexes(moead) == expectedSolutionIndexes);
        EXPECT_TRUE(GetExternalSolutionIndexes(moead) == expectedExternalSolutionIndexes);
    } else if (GetRank(moead) == 3) {
        expectedSolutionIndexes = {7, 8};
        expectedExternalSolutionIndexes = {6};
        EXPECT_TRUE(GetSolutionIndexes(moead) == expectedSolutionIndexes);
        EXPECT_TRUE(GetExternalSolutionIndexes(moead) == expectedExternalSolutionIndexes);
    }

    auto individuals = GetIndividuals(moead);
    EXPECT_EQ(individuals.size(), expectedSolutionIndexes.size() + expectedExternalSolutionIndexes.size());
    for (auto&& i : expectedSolutionIndexes) {
        expectedNeighborhood.clear();
        expectedNeighborhood.push_back(i);
        if (i == 0) {
            expectedNeighborhood.push_back(1);
            expectedNeighborhood.push_back(2);

        } else if (i == 8) {
            expectedNeighborhood.push_back(7);
            expectedNeighborhood.push_back(6);
        } else {
            expectedNeighborhood.push_back(i - 1);
            expectedNeighborhood.push_back(i + 1);
        }
        EXPECT_EQ(individuals[i].neighborhood, expectedNeighborhood);
    }
    for (auto&& i : expectedExternalSolutionIndexes) {
        EXPECT_TRUE(individuals[i].neighborhood.empty());
    }

    auto weightVectors = GetWeightVectors(moead);
    EXPECT_EQ(weightVectors.size(), expectedSolutionIndexes.size() + expectedExternalSolutionIndexes.size());
    int start = GetRank(moead) * 2;
    for (int i = start; i < start + weightVectors.size(); i++) {
        Eigen::ArrayXd wv(2);
        wv << (double)i / H, 1.0 - (double)i / H;
        EXPECT_TRUE((weightVectors[i] == wv).all());
    }
}

TEST_F(MpMoeadMpiTest, ScatterExternalNeighborhood) {
    auto moead = MpMoead<int>(5, 0, 0, 2, 0);
    moead.InitializeMpi(0, nullptr);
    std::vector<int> neighborhoodIndexes = {0, 1, 2, 3, 4};
    std::vector<int> neighborhoodSizes = {2, 1, 1, 1};
    std::vector<double> weightVectors = {0.0, 0.1,  //
                                         0.2, 0.3,  //
                                         0.4, 0.5,  //
                                         0.6, 0.7,  //
                                         0.8, 0.9};

    std::vector<int> receivedNeighborhoodIndexes;
    std::vector<double> receivedWeightVectors;
    std::tie(receivedNeighborhoodIndexes, receivedWeightVectors) =
        ScatterExternalNeighborhood(moead, neighborhoodIndexes, neighborhoodSizes, weightVectors);

    std::vector<int> expectedNeighborhoodIndexes;
    std::vector<double> expectedWeightVectors;
    if (GetRank(moead) == 0) {
        expectedNeighborhoodIndexes = {0, 1};
        expectedWeightVectors = {0.0, 0.1, 0.2, 0.3};
        EXPECT_EQ(receivedNeighborhoodIndexes, expectedNeighborhoodIndexes);
        EXPECT_EQ(receivedWeightVectors, expectedWeightVectors);
    } else if (GetRank(moead) == 1) {
        expectedNeighborhoodIndexes = {2};
        expectedWeightVectors = {0.4, 0.5};
        EXPECT_EQ(receivedNeighborhoodIndexes, expectedNeighborhoodIndexes);
        EXPECT_EQ(receivedWeightVectors, expectedWeightVectors);
    } else if (GetRank(moead) == 2) {
        expectedNeighborhoodIndexes = {3};
        expectedWeightVectors = {0.6, 0.7};
        EXPECT_EQ(receivedNeighborhoodIndexes, expectedNeighborhoodIndexes);
        EXPECT_EQ(receivedWeightVectors, expectedWeightVectors);
    } else if (GetRank(moead) == 3) {
        expectedNeighborhoodIndexes = {4};
        expectedWeightVectors = {0.8, 0.9};
        EXPECT_EQ(receivedNeighborhoodIndexes, expectedNeighborhoodIndexes);
        EXPECT_EQ(receivedWeightVectors, expectedWeightVectors);
    }
}

}  // namespace Eacpp::Test