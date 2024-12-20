#define _TEST_

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <eigen3/Eigen/Core>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "Algorithms/MpMoead.h"
#include "Crossovers/MockCrossover.h"
#include "Decompositions/MockDecomposition.h"
#include "Mutations/MockMutation.h"
#include "Problems/IProblem.h"
#include "Problems/MockProblem.h"
#include "Samplings/ISampling.h"
#include "Samplings/MockSampling.h"
#include "Selections/MockSelection.h"
#include "Utils/Utils.h"

using ::testing::_;
using ::testing::Return;

namespace Eacpp {

class MpMoeadTest : public ::testing::Test {
   protected:
    std::shared_ptr<MockCrossover<int>> mockCrossover;
    std::shared_ptr<MockDecomposition> mockDecomposition;
    std::shared_ptr<MockMutation<int>> mockMutation;
    std::shared_ptr<MockProblem<int>> mockProblem;
    std::shared_ptr<MockSampling<int>> mockSampling;
    std::shared_ptr<MockSelection> mockSelection;

    void SetUp() override {
        mockCrossover = std::make_shared<MockCrossover<int>>();
        mockDecomposition = std::make_shared<MockDecomposition>();
        mockMutation = std::make_shared<MockMutation<int>>();
        mockProblem = std::make_shared<MockProblem<int>>();
        mockSampling = std::make_shared<MockSampling<int>>();
        mockSelection = std::make_shared<MockSelection>();
    }

    // getter and setter
    template <typename T>
    std::unordered_map<int, typename MpMoead<T>::Individual> GetIndividuals(
        MpMoead<T>& moead) {
        return moead.individuals;
    }
    template <typename T>
    std::unordered_map<int, Eigen::ArrayXd> GetWeightVectors(
        MpMoead<T>& moead) {
        return moead.weightVectors;
    }
    template <typename T>
    Eigen::ArrayXd GetIdealPoint(MpMoead<T>& moead) {
        return moead.idealPoint;
    }
    template <typename T>
    std::vector<int> GetUpdatedSolutionIndexes(MpMoead<T>& moead) {
        return moead.updatedSolutionIndexes;
    }
    template <typename T>
    void SetSolutionIndexes(MpMoead<T>& moead,
                            std::vector<int> solutionIndexes) {
        moead.internalIndexes = solutionIndexes;
    }
    template <typename T>
    void SetExternalSolutionIndexes(MpMoead<T>& moead,
                                    std::vector<int> externalSolutionIndexes) {
        moead.externalIndexes = externalSolutionIndexes;
    }
    template <typename T>
    void SetIndividuals(MpMoead<T>& moead, int size) {
        for (int i = 0; i < size; i++) {
            moead.individuals[i] = typename MpMoead<T>::Individual();
        }
    }
    template <typename T>
    void AddIndividual(MpMoead<T>& moead, int index, Eigen::ArrayX<T> solution,
                       Eigen::ArrayXd objectives,
                       std::vector<int> neighborhood) {
        moead.individuals[index] =
            typename MpMoead<T>::Individual(solution, objectives, neighborhood);
    }
    template <typename T>
    void SetIdealPoint(MpMoead<T>& moead, Eigen::ArrayXd idealPoint) {
        moead.idealPoint = idealPoint;
    }
    template <typename T>
    void AddWeightVector(MpMoead<T>& moead, int index,
                         Eigen::ArrayXd weightVector) {
        moead.weightVectors[index] = weightVector;
    }
    // 初期化
    template <typename T>
    std::vector<int> GenerateSolutionIndexes(MpMoead<T>& moead, int rank,
                                             int parallelSize) {
        moead.rank = rank;
        moead.parallelSize = parallelSize;
        return moead.GenerateInternalIndexes();
    }
    template <typename T>
    std::vector<std::vector<double>> GenerateWeightVectors(MpMoead<T>& moead,
                                                           int H) {
        return moead.GenerateWeightVectors(H);
    }
    template <typename T>
    std::vector<std::vector<std::pair<double, int>>>
    CalculateEuclideanDistanceBetweenEachWeightVector(
        MpMoead<T>& moead, std::vector<double>& weightVectors) {
        return moead.CalculateEuclideanDistanceBetweenEachWeightVector(
            weightVectors);
    }
    template <typename T>
    std::vector<int> CalculateNeighborhoodIndexes(
        MpMoead<T>& moead,
        std::vector<std::vector<std::pair<double, int>>>& euclideanDistances) {
        return moead.CalculateNeighborhoodIndexes(euclideanDistances);
    }
    template <typename T>
    std::vector<int> GenerateNeighborhoods(
        MpMoead<T>& moead, std::vector<double>& allWeightVectors) {
        return moead.GenerateNeighborhoods(allWeightVectors);
    }
    template <typename T>
    std::pair<std::vector<int>, std::vector<int>> GenerateExternalNeighborhood(
        MpMoead<T>& moead, std::vector<int>& neighborhoodIndexes,
        std::vector<int>& populationSizes) {
        return moead.GenerateExternalNeighborhood(neighborhoodIndexes,
                                                  populationSizes);
    }
    template <typename T>
    std::vector<double> GetWeightVectorsMatchingIndexes(
        MpMoead<T>& moead, std::vector<double>& weightVectors,
        std::vector<int>& indexes) {
        return moead.GetWeightVectorsMatchingIndexes(weightVectors, indexes);
    }
    template <typename T>
    void InitializeIndividualAndWeightVector(
        MpMoead<T>& moead, std::vector<Eigen::ArrayXd>& weightVectors,
        std::vector<std::vector<int>>& neighborhoodIndexes,
        std::vector<Eigen::ArrayXd>& externalNeighboringWeightVectors) {
        moead.InitializeIndividualAndWeightVector(
            weightVectors, neighborhoodIndexes,
            externalNeighboringWeightVectors);
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
    std::vector<Eigen::ArrayX<T>> SelectParents(MpMoead<T>& moead, int index) {
        return moead.SelectParents(index);
    }
    template <typename T>
    Eigen::ArrayX<T> GenerateNewIndividual(MpMoead<T>& moead, int index) {
        return moead.GenerateNewIndividual(index);
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
    void UpdateSolution(MpMoead<T>& moead, int index,
                        Eigen::ArrayX<T>& solution,
                        Eigen::ArrayXd& objectiveSet) {
        moead.UpdateSolution(index, solution, objectiveSet);
    }
    template <typename T>
    void UpdateNeighboringIndividuals(
        MpMoead<T>& moead, int index, Eigen::ArrayX<T>& solution,
        Eigen::ArrayXd& objectiveSet,
        std::unordered_map<int, typename MpMoead<T>::Individual>
            externalIndividualCopies) {
        moead.UpdateNeighboringIndividuals(index, solution, objectiveSet,
                                           externalIndividualCopies);
    }
};

}  // namespace Eacpp

namespace Eacpp::Test {

TEST_F(MpMoeadTest, GenerateSolutionIndexes) {
    int totalPopulationSize = 9;
    int parallelSize = 4;
    MpMoead<int> moead = MpMoead<int>(totalPopulationSize, 0, 0, 0, 0, 0, 0);

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
    int objectivesNum = 2;
    int H = 2;
    MpMoead<int> moead = MpMoead<int>(0, 0, 0, objectivesNum, 0, 0, 0);
    auto actual = GenerateWeightVectors(moead, H);

    int expectedSize = 3;
    EXPECT_EQ(actual.size(), expectedSize);

    std::vector<std::vector<double>> expected = {
        {0.0, 1.0}, {0.5, 0.5}, {1.0, 0.0}};
    for (int i = 0; i < actual.size(); i++) {
        for (int j = 0; j < actual[i].size(); j++) {
            EXPECT_EQ(actual[i].size(), objectivesNum);
            EXPECT_EQ(actual[i][j], expected[i][j]);
        }
    }
}

TEST_F(MpMoeadTest, CalculateEuclideanDistanceBetweenEachWeightVector) {
    int totalPopulationSize = 3;
    int objectivesNum = 2;
    MpMoead<int> moead =
        MpMoead<int>(totalPopulationSize, 0, 0, objectivesNum, 0, 0, 0);
    std::vector<double> weightVectors = {
        10.0, 5.0,   //
        1.0,  20.0,  //
        15.0, 10.0,
    };
    std::vector<std::vector<std::pair<double, int>>> expected = {
        {{0.0, 0}, {306.0, 1}, {50.0, 2}},   //
        {{306.0, 0}, {0.0, 1}, {296.0, 2}},  //
        {{50.0, 0}, {296.0, 1}, {0.0, 2}}};
    auto actual =
        CalculateEuclideanDistanceBetweenEachWeightVector(moead, weightVectors);
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
    int neighborhoodSize = 2;
    auto moead =
        MpMoead<int>(totalPopulationSize, 0, 0, 0, neighborhoodSize, 0, 0);
    std::vector<std::vector<std::pair<double, int>>> euclideanDistances = {
        {{0.0, 0}, {306.0, 1}, {50.0, 2}},   //
        {{306.0, 0}, {0.0, 1}, {296.0, 2}},  //
        {{50.0, 0}, {296.0, 1}, {0.0, 2}}};
    auto actual = CalculateNeighborhoodIndexes(moead, euclideanDistances);
    std::vector<int> expected = {0, 2, 1, 2, 2, 0};
    EXPECT_EQ(actual, expected);
}

TEST_F(MpMoeadTest, GenerateNeighborhoods) {
    int totalPopulationSize = 3;
    int objectivesNum = 2;
    int neighborhoodSize = 2;
    MpMoead<int> moead = MpMoead<int>(totalPopulationSize, 0, 0, objectivesNum,
                                      neighborhoodSize, 0, 0);
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
    int neighborhoodSize = 2;
    auto moead = MpMoead<int>(totalPopulationSize, 0, 0, 0, 2, 0, 0);
    std::vector<int> neighborhoodIndexes = {0, 1, 1, 0, 2, 3, 3, 2};
    std::vector<int> populationSizes = {2, 1, 1};
    auto actual = GenerateExternalNeighborhood(moead, neighborhoodIndexes,
                                               populationSizes);
    std::vector<int> expectedIndexes = {3, 2};
    std::vector<int> expectedSizes = {0, 1, 1};
    EXPECT_EQ(actual.first, expectedIndexes);
    EXPECT_EQ(actual.second, expectedSizes);
}

TEST_F(MpMoeadTest, GetWeightVectorsMatchingIndexes) {
    int objectivesNum = 2;
    auto moead = MpMoead<int>(0, 0, 0, objectivesNum, 0, 0, 0);
    std::vector<double> weightVectors = {
        1.0, 2.0,  //
        3.0, 4.0,  //
        5.0, 6.0,
    };
    std::vector<int> indexes = {0, 2};
    std::vector<double> expected = {1.0, 2.0, 5.0, 6.0};
    auto actual =
        GetWeightVectorsMatchingIndexes(moead, weightVectors, indexes);
    EXPECT_EQ(actual, expected);
}

TEST_F(MpMoeadTest, InitializeIndividualAndWeightVector) {
    auto moead = MpMoead<int>(0, 0, 0, 0, 0, 0, 0);
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
    std::vector<std::vector<int>> neighborhoodIndexes = {
        {0, 1}, {1, 2}, {2, 3}};
    std::vector<Eigen::ArrayXd> externalNeighboringWeightVectors;
    Eigen::ArrayXd wv(2);
    wv << 10.0, 11.0;
    externalNeighboringWeightVectors.push_back(wv);

    InitializeIndividualAndWeightVector(moead, weightVectors,
                                        neighborhoodIndexes,
                                        externalNeighboringWeightVectors);

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
        EXPECT_TRUE(
            (weightVectorsMap[i] == externalNeighboringWeightVectors[count])
                .all());
        count++;
    }
}

TEST_F(MpMoeadTest, InitializePopulation) {
    int decisionVariableNum = 2;
    int sampleNum = 2;

    Eigen::ArrayXi samplingResult = Eigen::ArrayXi::LinSpaced(
        decisionVariableNum, 0, decisionVariableNum - 1);
    std::vector<Eigen::ArrayXi> samplingResults(sampleNum, samplingResult);
    EXPECT_CALL(*mockSampling, Sample(_, _))
        .WillRepeatedly(Return(samplingResults));

    Eigen::ArrayXd objectiveSetResult = Eigen::ArrayXd::LinSpaced(2, 0, 1);
    EXPECT_CALL(*mockProblem, ComputeObjectiveSet(_))
        .WillRepeatedly(Return(objectiveSetResult));

    auto moead =
        MpMoead<int>(0, 0, decisionVariableNum, 0, 0, 0, 0, nullptr, nullptr,
                     nullptr, mockProblem, mockSampling, nullptr);
    SetIndividuals(moead, sampleNum);
    InitializePopulation(moead);

    auto individuals = GetIndividuals(moead);
    for (auto&& i : individuals) {
        EXPECT_TRUE((i.second.solution == samplingResult).all());
        EXPECT_TRUE((i.second.objectives == objectiveSetResult).all());
    }
}

TEST_F(MpMoeadTest, InitializeIdealPoint) {
    auto moead = MpMoead<int>(0, 0, 0, 0, 0, 0, 0);
    AddIndividual(moead, 0, Eigen::ArrayXi(),
                  Eigen::ArrayXd::LinSpaced(2, 0, 1), {});
    Eigen::ArrayXd expected = Eigen::ArrayXd::LinSpaced(2, 0, 1);

    InitializeIdealPoint(moead);
    auto actual = GetIdealPoint(moead);

    EXPECT_TRUE((actual == expected).all());

    Eigen::ArrayXd objectives = Eigen::ArrayXd(2);
    objectives << 1.0, 0.0;
    AddIndividual(moead, 1, Eigen::ArrayXi(), objectives, {});
    expected << 0.0, 0.0;

    InitializeIdealPoint(moead);
    actual = GetIdealPoint(moead);

    EXPECT_TRUE((actual == expected).all());
}

TEST_F(MpMoeadTest, SelectParents) {
    EXPECT_CALL(*mockSelection, Select(_, _))
        .WillRepeatedly(Return(std::vector<int>{1}));

    EXPECT_CALL(*mockCrossover, GetParentNum()).WillRepeatedly(Return(2));

    auto moead = MpMoead<int>(0, 0, 0, 0, 0, 0, 0, mockCrossover, nullptr,
                              nullptr, nullptr, nullptr, mockSelection);
    Eigen::ArrayXi solution0 = Eigen::ArrayXi::LinSpaced(2, 0, 1);
    Eigen::ArrayXi solution1 = Eigen::ArrayXi::LinSpaced(2, 1, 2);
    AddIndividual(moead, 0, solution0, Eigen::ArrayXd(), {1});
    AddIndividual(moead, 1, solution1, Eigen::ArrayXd(), {0});

    auto actual = SelectParents(moead, 0);
    std::vector<Eigen::ArrayXi> expected = {solution0, solution1};
    for (int i = 0; i < actual.size(); i++) {
        EXPECT_TRUE((actual[i] == expected[i]).all());
    }
}

TEST_F(MpMoeadTest, GenerateNewIndividual) {
    EXPECT_CALL(*mockSelection, Select(_, _))
        .WillRepeatedly(Return(std::vector<int>{1}));

    Eigen::ArrayXi expected = Eigen::ArrayXi::LinSpaced(2, 10, 11);
    EXPECT_CALL(*mockCrossover, GetParentNum()).WillRepeatedly(Return(2));
    EXPECT_CALL(*mockCrossover, Cross(_)).WillOnce(Return(expected));

    EXPECT_CALL(*mockMutation, Mutate(_)).Times(1);

    auto moead = MpMoead<int>(0, 0, 0, 0, 0, 0, 0, mockCrossover, nullptr,
                              mockMutation, nullptr, nullptr, mockSelection);
    Eigen::ArrayXi solution0 = Eigen::ArrayXi::LinSpaced(2, 0, 1);
    Eigen::ArrayXi solution1 = Eigen::ArrayXi::LinSpaced(2, 1, 2);
    AddIndividual(moead, 0, solution0, Eigen::ArrayXd(), {1});
    AddIndividual(moead, 1, solution1, Eigen::ArrayXd(), {0});

    auto actual = GenerateNewIndividual(moead, 0);

    EXPECT_TRUE((actual == expected).all());
}

TEST_F(MpMoeadTest, RepairSolution) {
    EXPECT_CALL(*mockProblem, IsFeasible(_))
        .WillOnce(Return(true))
        .WillOnce(Return(false));

    Eigen::ArrayXi expected2 = Eigen::ArrayXi::LinSpaced(2, 10, 11);
    EXPECT_CALL(*mockSampling, Sample(_, _))
        .WillRepeatedly(Return(std::vector<Eigen::ArrayXi>{expected2}));

    auto moead = MpMoead<int>(0, 0, 0, 0, 0, 0, 0, nullptr, nullptr, nullptr,
                              mockProblem, mockSampling, nullptr);
    Eigen::ArrayXi solution = Eigen::ArrayXi::LinSpaced(2, 0, 1);
    auto expected1 = solution;

    RepairSolution(moead, solution);
    EXPECT_TRUE((solution == expected1).all());

    RepairSolution(moead, solution);
    EXPECT_TRUE((solution == expected2).all());
}

TEST_F(MpMoeadTest, UpdateIdealPoint) {
    auto moead = MpMoead<int>(0, 0, 0, 0, 0, 0, 0);
    Eigen::ArrayXd idealPoint = Eigen::ArrayXd::LinSpaced(2, 1, 2);
    SetIdealPoint(moead, idealPoint);

    Eigen::ArrayXd objectiveSet = Eigen::ArrayXd::LinSpaced(2, 0, 1);

    UpdateIdealPoint(moead, objectiveSet);

    auto actual = GetIdealPoint(moead);
    EXPECT_TRUE((actual == objectiveSet).all());
}

TEST_F(MpMoeadTest, UpdateSolution) {
    EXPECT_CALL(*mockDecomposition, ComputeObjective(_, _, _))
        .WillOnce(Return(1.0))
        .WillOnce(Return(0.0))
        .WillOnce(Return(0.0))
        .WillOnce(Return(1.0));
    auto moead = MpMoead<int>(0, 0, 0, 0, 0, 0, 0, nullptr, mockDecomposition,
                              nullptr, nullptr, nullptr, nullptr);
    int index = 0;
    SetIdealPoint(moead, Eigen::ArrayXd());
    AddWeightVector(moead, index, Eigen::ArrayXd());

    Eigen::ArrayXi indSolution = Eigen::ArrayXi::LinSpaced(2, 0, 1);
    Eigen::ArrayXd indObjectives = Eigen::ArrayXd::LinSpaced(2, 0, 1);
    AddIndividual(moead, index, indSolution, indObjectives, {});
    Eigen::ArrayXi solution = Eigen::ArrayXi::LinSpaced(2, 1, 2);
    Eigen::ArrayXd objectives = Eigen::ArrayXd::LinSpaced(2, 1, 2);

    UpdateSolution(moead, index, solution, objectives);

    auto individuals = GetIndividuals(moead);
    auto updatedSolutionIndexes = GetUpdatedSolutionIndexes(moead);
    EXPECT_TRUE((individuals[index].solution == indSolution).all());
    EXPECT_TRUE((individuals[index].objectives == indObjectives).all());
    EXPECT_TRUE(updatedSolutionIndexes.empty());

    UpdateSolution(moead, index, solution, objectives);

    individuals = GetIndividuals(moead);
    updatedSolutionIndexes = GetUpdatedSolutionIndexes(moead);
    EXPECT_TRUE((individuals[index].solution == solution).all());
    EXPECT_TRUE((individuals[index].objectives == objectives).all());
    EXPECT_EQ(updatedSolutionIndexes.size(), 1);
    EXPECT_EQ(updatedSolutionIndexes[0], index);
}

TEST_F(MpMoeadTest, UpdateNeighboringIndividuals) {}

}  // namespace Eacpp::Test