#define _TEST_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <eigen3/Eigen/Core>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "Algorithms/MpMoead.h"
#include "Problems/IProblem.h"
#include "Problems/MockProblem.h"
#include "Samplings/ISampling.h"
#include "Samplings/MockSampling.h"
#include "Utils/Utils.h"

using ::testing::_;
using ::testing::Return;

namespace Eacpp {

class MpMoeadTest : public ::testing::Test {
   protected:
    // getter and setter
    template <typename T>
    std::unordered_map<int, typename MpMoead<T>::Individual> GetIndividuals(MpMoead<T>& moead) {
        return moead.individuals;
    }
    template <typename T>
    std::unordered_map<int, Eigen::ArrayXd> GetWeightVectors(MpMoead<T>& moead) {
        return moead.weightVectors;
    }
    template <typename T>
    void SetSolutionIndexes(MpMoead<T>& moead, std::vector<int> solutionIndexes) {
        moead.solutionIndexes = solutionIndexes;
    }
    template <typename T>
    void SetExternalSolutionIndexes(MpMoead<T>& moead, std::vector<int> externalSolutionIndexes) {
        moead.externalSolutionIndexes = externalSolutionIndexes;
    }
    template <typename T>
    void SetIndividuals(MpMoead<T>& moead, int size) {
        for (int i = 0; i < size; i++) {
            moead.individuals[i] = typename MpMoead<T>::Individual();
        }
    }
    // 初期化
    template <typename T>
    std::vector<int> GenerateSolutionIndexes(MpMoead<T>& moead, int rank, int parallelSize) {
        moead.rank = rank;
        moead.parallelSize = parallelSize;
        return moead.GenerateSolutionIndexes();
    }
    template <typename T>
    std::vector<std::vector<double>> GenerateWeightVectors(MpMoead<T>& moead, int H) {
        return moead.GenerateWeightVectors(H);
    }
    template <typename T>
    std::vector<std::vector<std::pair<double, int>>> CalculateEuclideanDistanceBetweenEachWeightVector(
        MpMoead<T>& moead, std::vector<double>& weightVectors) {
        return moead.CalculateEuclideanDistanceBetweenEachWeightVector(weightVectors);
    }
    template <typename T>
    std::vector<int> CalculateNeighborhoodIndexes(MpMoead<T>& moead,
                                                  std::vector<std::vector<std::pair<double, int>>>& euclideanDistances) {
        return moead.CalculateNeighborhoodIndexes(euclideanDistances);
    }
    template <typename T>
    std::vector<int> GenerateNeighborhoods(MpMoead<T>& moead, std::vector<double>& allWeightVectors) {
        return moead.GenerateNeighborhoods(allWeightVectors);
    }
    template <typename T>
    std::pair<std::vector<int>, std::vector<int>> GenerateExternalNeighborhood(MpMoead<T>& moead,
                                                                               std::vector<int>& neighborhoodIndexes,
                                                                               std::vector<int>& populationSizes) {
        return moead.GenerateExternalNeighborhood(neighborhoodIndexes, populationSizes);
    }
    template <typename T>
    std::vector<double> GetWeightVectorsMatchingIndexes(MpMoead<T>& moead, std::vector<double>& weightVectors,
                                                        std::vector<int>& indexes) {
        return moead.GetWeightVectorsMatchingIndexes(weightVectors, indexes);
    }
    template <typename T>
    void InitializeIndividualAndWeightVector(MpMoead<T>& moead, std::vector<Eigen::ArrayXd>& weightVectors,
                                             std::vector<std::vector<int>>& neighborhoodIndexes,
                                             std::vector<Eigen::ArrayXd>& externalNeighboringWeightVectors) {
        moead.InitializeIndividualAndWeightVector(weightVectors, neighborhoodIndexes, externalNeighboringWeightVectors);
    }
    // Update
    template <typename T>
    void InitializePopulation(MpMoead<T>& moead) {
        moead.InitializePopulation();
    }
    template <typename T>
    void InitializeIdealPoint(MpMoead<T>& moead) {
        moead.InitializeIdealPoint();
    }
    template <typename T>
    Eigen::ArrayX<T> GenerateNewSolution(MpMoead<T>& moead, int index) {
        return moead.GenerateNewSolution(index);
    }
    template <typename T>
    void RepairSolution(MpMoead<T>& moead, Eigen::ArrayX<T>& solution) {
        moead.RepairSolution(solution);
    }
    template <typename T>
    void UpdateIdealPoint(MpMoead<T>& mead, Eigen::ArrayXd& objectiveSet) {
        mead.UpdateIdealPoint(objectiveSet);
    }
    template <typename T>
    void UpdateSolution(MpMoead<T>& moead, int index, Eigen::ArrayX<T>& solution, Eigen::ArrayXd& objectiveSet) {
        moead.UpdateSolution(index, solution, objectiveSet);
    }
    template <typename T>
    void UpdateNeighboringSolutions(MpMoead<T>& moead, int index, Eigen::ArrayX<T>& solution, Eigen::ArrayXd& objectiveSet,
                                    std::unordered_map<int, typename MpMoead<T>::Individual> externalIndividualCopies) {
        moead.UpdateNeighboringSolutions(index, solution, objectiveSet, externalIndividualCopies);
    }
};

}  // namespace Eacpp

namespace Eacpp::Test {

TEST_F(MpMoeadTest, GenerateSolutionIndexes) {
    int totalPopulationSize = 9;
    int parallelSize = 4;
    MpMoead<int> moead = MpMoead<int>(totalPopulationSize, 0, 0, 0, 0);

    auto actual = GenerateSolutionIndexes(moead, 0, parallelSize);
    std::vector<int> expected = {0, 1, 2};
    EXPECT_TRUE(actual == expected);

    actual = GenerateSolutionIndexes(moead, 1, parallelSize);
    expected = {3, 4};
    EXPECT_TRUE(actual == expected);

    actual = GenerateSolutionIndexes(moead, 3, parallelSize);
    expected = {7, 8};
    EXPECT_TRUE(actual == expected);
}

TEST_F(MpMoeadTest, GenerateWeightVectors) {
    int objectiveNum = 2;
    int H = 2;
    MpMoead<int> moead = MpMoead<int>(0, 0, 0, objectiveNum, 0);
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
    int totalPopulationSize = 3;
    int objectiveNum = 2;
    MpMoead<int> moead = MpMoead<int>(totalPopulationSize, 0, 0, objectiveNum, 0);
    std::vector<double> weightVectors = {
        10.0, 5.0,   //
        1.0,  20.0,  //
        15.0, 10.0,
    };
    std::vector<std::vector<std::pair<double, int>>> expected = {{{0.0, 0}, {306.0, 1}, {50.0, 2}},   //
                                                                 {{306.0, 0}, {0.0, 1}, {296.0, 2}},  //
                                                                 {{50.0, 0}, {296.0, 1}, {0.0, 2}}};
    auto actual = CalculateEuclideanDistanceBetweenEachWeightVector(moead, weightVectors);
    for (int i = 0; i < actual.size(); i++) {
        for (int j = 0; j < actual[i].size(); j++) {
            EXPECT_EQ(actual[i].size(), totalPopulationSize);
            EXPECT_EQ(actual[i][j].first, expected[i][j].first);
            EXPECT_EQ(actual[i][j].second, expected[i][j].second);
        }
    }
}

TEST_F(MpMoeadTest, CalculateNeighborhoodIndexes) {
    int totalPopulationSize = 3;
    int neighborNum = 2;
    auto moead = MpMoead<int>(totalPopulationSize, 0, 0, 0, neighborNum);
    std::vector<std::vector<std::pair<double, int>>> euclideanDistances = {{{0.0, 0}, {306.0, 1}, {50.0, 2}},   //
                                                                           {{306.0, 0}, {0.0, 1}, {296.0, 2}},  //
                                                                           {{50.0, 0}, {296.0, 1}, {0.0, 2}}};
    auto actual = CalculateNeighborhoodIndexes(moead, euclideanDistances);
    std::vector<int> expected = {0, 2, 1, 2, 2, 0};
    EXPECT_EQ(actual, expected);
}

TEST_F(MpMoeadTest, GenerateNeighborhoods) {
    int totalPopulationSize = 3;
    int objectiveNum = 2;
    int neighborNum = 2;
    MpMoead<int> moead = MpMoead<int>(totalPopulationSize, 0, 0, objectiveNum, neighborNum);
    std::vector<double> allWeightVectors = {
        10.0, 5.0,   //
        1.0,  20.0,  //
        15.0, 10.0,
    };
    std::vector<int> expected = {0, 2, 1, 2, 2, 0};
    auto actual = GenerateNeighborhoods(moead, allWeightVectors);
    EXPECT_EQ(actual, expected);
}

TEST_F(MpMoeadTest, GenerateExternalNeighborhood) {
    int totalPopulationSize = 4;
    int neighborNum = 2;
    auto moead = MpMoead<int>(totalPopulationSize, 0, 0, 0, 2);
    std::vector<int> neighborhoodIndexes = {0, 1, 1, 0, 2, 3, 3, 2};
    std::vector<int> populationSizes = {2, 1, 1};
    auto actual = GenerateExternalNeighborhood(moead, neighborhoodIndexes, populationSizes);
    std::vector<int> expectedIndexes = {3, 2};
    std::vector<int> expectedSizes = {0, 1, 1};
    EXPECT_EQ(actual.first, expectedIndexes);
    EXPECT_EQ(actual.second, expectedSizes);
}

TEST_F(MpMoeadTest, GetWeightVectorsMatchingIndexes) {
    int objectiveNum = 2;
    auto moead = MpMoead<int>(0, 0, 0, objectiveNum, 0);
    std::vector<double> weightVectors = {
        1.0, 2.0,  //
        3.0, 4.0,  //
        5.0, 6.0,
    };
    std::vector<int> indexes = {0, 2};
    std::vector<double> expected = {1.0, 2.0, 5.0, 6.0};
    auto actual = GetWeightVectorsMatchingIndexes(moead, weightVectors, indexes);
    EXPECT_EQ(actual, expected);
}

TEST_F(MpMoeadTest, InitializeIndividualAndWeightVector) {
    auto moead = MpMoead<int>(0, 0, 0, 0, 0);
    std::vector<int> solutionIndexes = {0, 1, 2};
    std::vector<int> externalSolutionIndexes = {3};
    SetSolutionIndexes(moead, solutionIndexes);
    SetExternalSolutionIndexes(moead, externalSolutionIndexes);
    std::vector<Eigen::ArrayXd> weightVectors(3);
    for (int i = 0; i < weightVectors.size(); i++) {
        Eigen::ArrayXd wv(2);
        wv << (double)i, (double)(i + 1);
        weightVectors[i] = wv;
    }
    std::vector<std::vector<int>> neighborhoodIndexes = {{0, 1}, {1, 2}, {2, 3}};
    std::vector<Eigen::ArrayXd> externalNeighboringWeightVectors;
    Eigen::ArrayXd wv(2);
    wv << 10.0, 11.0;
    externalNeighboringWeightVectors.push_back(wv);

    InitializeIndividualAndWeightVector(moead, weightVectors, neighborhoodIndexes, externalNeighboringWeightVectors);

    auto individuals = GetIndividuals(moead);
    EXPECT_EQ(individuals.size(), 4);
    for (int count = 0; auto&& i : solutionIndexes) {
        EXPECT_EQ(individuals[i].neighborhood, neighborhoodIndexes[count]);
        count++;
    }
    for (auto&& i : externalSolutionIndexes) {
        EXPECT_EQ(individuals[i].neighborhood.size(), 0);
    }

    auto weightVectorsMap = GetWeightVectors(moead);
    EXPECT_EQ(weightVectorsMap.size(), 4);
    for (int count = 0; auto&& i : solutionIndexes) {
        EXPECT_TRUE((weightVectorsMap[i] == weightVectors[count]).all());
        count++;
    }
    for (int count = 0; auto&& i : externalSolutionIndexes) {
        EXPECT_TRUE((weightVectorsMap[i] == externalNeighboringWeightVectors[count]).all());
        count++;
    }
}

TEST_F(MpMoeadTest, InitializePopulation) {
    int decisionVariableNum = 2;
    int sampleNum = 2;

    auto mockSampling = std::make_shared<MockSampling<int>>();
    // Eigen::ArrayXi samplingResult = Eigen::ArrayXi::LinSpaced(decisionVariableNum, 0, decisionVariableNum - 1);
    // std::vector<Eigen::ArrayXi> samplingResults(sampleNum, samplingResult);
    // EXPECT_CALL(*mockSampling, Sample(_, _)).WillRepeatedly(Return(samplingResults));

    // std::shared_ptr<MockProblem<int>> mockProblem = std::make_shared<MockProblem<int>>();
    // Eigen::ArrayXd objectiveSetResult = Eigen::ArrayXd::LinSpaced(2, 0, 1);
    // EXPECT_CALL(*mockProblem, ComputeObjectiveSet(_)).WillRepeatedly(Return(objectiveSetResult));

    // auto moead = MpMoead<int>(0, 0, decisionVariableNum, 0, 0, nullptr, nullptr, nullptr, nullptr, mockSampling, nullptr);
    // SetIndividuals(moead, sampleNum);
    // InitializePopulation(moead);

    // auto individuals = GetIndividuals(moead);
    // for (auto&& i : individuals) {
    //     EXPECT_TRUE((i.second.solution == samplingResult).all());
    //     EXPECT_TRUE((i.second.objectives == objectiveSetResult).all());
    // }
}

TEST_F(MpMoeadTest, InitializeIdealPoint) {}

TEST_F(MpMoeadTest, GenerateNewSolution) {}

TEST_F(MpMoeadTest, RepairSolution) {}

TEST_F(MpMoeadTest, UpdateIdealPoint) {}

TEST_F(MpMoeadTest, UpdateSolution) {}

TEST_F(MpMoeadTest, UpdateNeighboringSolutions) {}

}  // namespace Eacpp::Test