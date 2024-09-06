#pragma once

#include <mpi.h>

#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <memory>
#include <numeric>
#include <ranges>
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "Crossovers/ICrossover.h"
#include "Decompositions/IDecomposition.h"
#include "Mutations/IMutation.h"
#include "Problems/IProblem.h"
#include "Samplings/ISampling.h"
#include "Selections/ISelection.h"
#include "Utils/MpiUtils.h"
#include "Utils/TemplateType.h"
#include "Utils/Utils.h"

namespace Eacpp {

template <typename DecisionVariableType>
class MpMoead {
   public:
    int generationNum;
    int populationSize;
    int decisionVariableNum;
    int objectiveNum;
    int neighborNum;
    std::shared_ptr<ICrossover<DecisionVariableType>> crossover;
    std::shared_ptr<IDecomposition> decomposition;
    std::shared_ptr<IMutation<DecisionVariableType>> mutation;
    std::shared_ptr<IProblem<DecisionVariableType>> problem;
    std::shared_ptr<ISampling<DecisionVariableType>> sampling;
    std::shared_ptr<ISelection> selection;
    std::vector<Eigen::ArrayXd> weightVectors;
    std::vector<Eigen::ArrayX<DecisionVariableType>> solutions;
    std::vector<Eigen::ArrayXd> objectiveSets;
    Eigen::ArrayXd idealPoint;
    std::vector<std::vector<int>> neighborhoodIndexes;
    std::vector<Eigen::ArrayX<DecisionVariableType>> allNeighborSolutions;
    std::vector<Eigen::ArrayXd> allNeighborObjectiveSets;
    int rank;
    int parallelSize;
    std::vector<std::vector<int>> externalNeighborhoodIndexes;
    std::vector<Eigen::ArrayXd> externalNeighboringWeightVectors;
    std::vector<int> solutionIndexes;

    MpMoead(int generationNum, int decisionVariableNum, int objectiveNum, int neighborNum,
            std::shared_ptr<ICrossover<DecisionVariableType>> crossover, std::shared_ptr<IDecomposition> decomposition,
            std::shared_ptr<IMutation<DecisionVariableType>> mutation, std::shared_ptr<IProblem<DecisionVariableType>> problem,
            std::shared_ptr<ISampling<DecisionVariableType>> sampling, std::shared_ptr<ISelection> selection)
        : generationNum(generationNum),
          decisionVariableNum(decisionVariableNum),
          objectiveNum(objectiveNum),
          neighborNum(neighborNum),
          crossover(crossover),
          decomposition(decomposition),
          mutation(mutation),
          problem(problem),
          sampling(sampling),
          selection(selection) {}
    virtual ~MpMoead() {}

    void Run(int argc, char** argv);
    void Initialize(int totalPopulationSize, int H);
    void InitializeIsland();
    void Update();

   private:
    void InitializeMpi(int argc, char** argv);
    std::vector<int> GenerateSolutionIndexes(int totalPopulationSize);
    std::vector<std::vector<double>> GenerateWeightVectors(int H);
    std::vector<int> GenerateNeighborhoods(int totalPopulationSize, std::vector<double>& allWeightVectors);
    std::vector<std::vector<std::pair<double, int>>> CalculateEuclideanDistanceBetweenEachWeightVector(
        int totalPopulationSize, std::vector<double>& weightVectors);
    std::vector<int> CalculateNeighborhoodIndexes(int totalPopulationSize,
                                                  std::vector<std::vector<std::pair<double, int>>>& euclideanDistances);

    // template <typename T>
    // std::vector<T> Scatter(std::vector<T>& data, std::vector<int>& populationSizes, int singleDataSize);

    std::tuple<std::vector<int>, std::vector<int>> GenerateExternalNeighbourhood(std::vector<int>& neighborhoodIndexes,
                                                                                 std::vector<int>& populationSizes);
    std::vector<double> GetWeightVectorMatchingIndex(std::vector<double> weightVectors, std::vector<int> indexes);
    std::pair<std::vector<int>, std::vector<double>> ScatterExternalNeighborhood(std::vector<int>& neighborhoodIndexes,
                                                                                 std::vector<int>& neighborhoodSizes,
                                                                                 std::vector<double>& weightVectors);

    void InitializePopulation();
    void InitializeIdealPoint();
    Eigen::ArrayX<DecisionVariableType> GenerateNewSolution(int index);
    void UpdateIdealPoint(Eigen::ArrayXd objectiveSet);
    void UpdateNeighboringSolutions(int index, Eigen::ArrayX<DecisionVariableType> solution, Eigen::ArrayXd objectiveSet);

   public:
#ifdef _TEST_
    friend class MpMpeadTest;
#endif
};

// template <typename DecisionVariableType>
// void MpMoead<DecisionVariableType>::Run(int argc, char** argv) {
//     MPI_Init(&argc, &argv);
//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     if (rank == 0) {
//         Initialize(100, rank, size);
//     }

//     // input
//     // rank0から近傍を送信

//     for (int i = 0; i < generationNum; i++) {
//         Update();
//     }

//     // Finalize MPI
//     MPI_Finalize();
// }

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::InitializeMpi(int argc, char** argv) {
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(&argc, &argv);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &parallelSize);
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::Initialize(int totalPopulationSize, int H) {
    std::vector<double> weightVectors1d;
    std::vector<int> allNeighborhoodIndexes;
    if (rank == 0) {
        std::vector<std::vector<double>> weightVectors2d = GenerateWeightVectors(H);
        weightVectors1d = TransformTo1d(weightVectors2d);
        allNeighborhoodIndexes = GenerateNeighborhoods(totalPopulationSize, weightVectors1d);
    }

    populationSize = CalculateNodeWorkload(totalPopulationSize, rank, parallelSize);

    std::vector<int> populationSizes;
    if (rank == 0) {
        populationSizes = CalculateNodeWorkloads(totalPopulationSize, parallelSize);
    }

    std::vector<double> receivedWeightVectors = Scatterv(weightVectors1d, populationSizes, objectiveNum, rank, parallelSize);
    weightVectors = TransformToEigenArrayX2d(receivedWeightVectors, objectiveNum);

    std::vector<int> receivedNeighborhoodIndexes =
        Scatterv(allNeighborhoodIndexes, populationSizes, neighborNum, rank, parallelSize);
    neighborhoodIndexes = TransformTo2d(receivedNeighborhoodIndexes, neighborNum);

    std::vector<int> unduplicatedNeighborhoodIndexes;
    std::vector<int> neighborhoodSizes;
    std::vector<double> sendExternalNeighboringWeightVectors;
    if (rank == 0) {
        std::tie(unduplicatedNeighborhoodIndexes, neighborhoodSizes) =
            GenerateExternalNeighbourhood(allNeighborhoodIndexes, populationSizes);
        sendExternalNeighboringWeightVectors = GetWeightVectorMatchingIndex(weightVectors1d, unduplicatedNeighborhoodIndexes);
    }
    std::vector<double> receivedExternalNeighboringWeightVectors;
    std::tie(externalNeighborhoodIndexes, receivedExternalNeighboringWeightVector) =
        ScatterExternalNeighborhood(unduplicatedNeighborhoodIndexes, neighborhoodSizes, sendExternalNeighboringWeightVectors);
    externalNeighboringWeightVectors = TransformToEigenArrayX2d(receivedExternalNeighboringWeightVectors, objectiveNum);
}

// template <typename DecisionVariableType>
// void MpMoead<DecisionVariableType>::InitializeIsland() {
//     allNeighborSolutions.reserve(populationSize);
//     allNeighborObjectiveSets.reserve(populationSize);
//     for (int i = 0; i < populationSize; i++) {
//         std::vector<Eigen::ArrayX<DecisionVariableType>> neighborSolutions;
//         std::vector<Eigen::ArrayXd> neighborObjectiveSets;
//         for (int j = 0; j < neighborNum; j++) {
//             neighborSolutions.push_back(sampling->Sample(1, decisionVariableNum)[0]);
//             neighborObjectiveSets.push_back(problem->ComputeObjectiveSet(neighborSolutions[j]));
//         }
//         neighborSolutions.push_back(neighborSolutions);
//         neighborObjectiveSets.push_back(neighborObjectiveSets);
//     }
// }

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::Update() {}

// template <typename DecisionVariableType>
// void MpMoead<DecisionVariableType>::Update() {
//     for (int i = 0; i < populationSize; i++) {
//         Eigen::ArrayX<DecisionVariableType> newSolution = GenerateNewSolution(i);
//         if (!problem->IsFeasible(newSolution)) {
//             newSolution = sampling->Sample(1, decisionVariableNum)[0];
//         }
//         Eigen::ArrayXd newObjectiveSet = problem->ComputeObjectiveSet(newSolution);
//         UpdateIdealPoint(newObjectiveSet);
//         UpdateNeighboringSolutions(i, newSolution, newObjectiveSet);
//     }
// }

template <typename DecisionVariableType>
std::vector<int> MpMoead<DecisionVariableType>::GenerateSolutionIndexes(int totalPopulationSize) {
    int start = CalculateNodeStartIndex(totalPopulationSize, rank, parallelSize);
    std::vector<int> solutionIndexes = Rangei(start, start + populationSize - 1, 1);
    return solutionIndexes;
}

template <typename DecisionVariableType>
std::vector<std::vector<double>> MpMoead<DecisionVariableType>::GenerateWeightVectors(int H) {
    std::vector<double> takedSetTop(H + 1);
    std::iota(takedSetTop.begin(), takedSetTop.end(), 0);
    std::vector<std::vector<double>> product = Product(takedSetTop, objectiveNum);
    product.erase(std::remove_if(product.begin(), product.end(), [&](auto v) { return std::reduce(v.begin(), v.end()) != H; }),
                  product.end());
    return product;
}

template <typename DecisionVariableType>
std::vector<int> MpMoead<DecisionVariableType>::GenerateNeighborhoods(int totalPopulationSize,
                                                                      std::vector<double>& allWeightVectors) {
    std::vector<std::vector<std::pair<double, int>>> euclideanDistances =
        CalculateEuclideanDistanceBetweenEachWeightVector(totalPopulationSize, allWeightVectors);
    std::vector<int> neighborhoodIndexes = CalculateNeighborhoodIndexes(totalPopulationSize, euclideanDistances);
    return neighborhoodIndexes;
}

template <typename DecisionVariableType>
std::vector<std::vector<std::pair<double, int>>>
MpMoead<DecisionVariableType>::CalculateEuclideanDistanceBetweenEachWeightVector(int totalPopulationSize,
                                                                                 std::vector<double>& weightVectors) {
    std::vector<std::vector<std::pair<double, int>>> euclideanDistances(
        totalPopulationSize, std::vector<std::pair<double, int>>(totalPopulationSize));
    for (int i = 0; i < totalPopulationSize; i++) {
        for (int j = 0; j < totalPopulationSize; j++) {
            std::vector<double> diff(objectiveNum);
            for (int k = 0; k < objectiveNum; k++) {
                diff[k] = allWeightVectors[i * objectiveNum + k] - allWeightVectors[j * objectiveNum + k];
            }
            double squaredNorm = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
            euclideanDistances[i][j] = std::make_pair(squaredNorm, j);
        }
    }
    return euclideanDistances;
}

template <typename DecisionVariableType>
std::vector<int> MpMoead<DecisionVariableType>::CalculateNeighborhoodIndexes(
    int totalPopulationSize, std::vector<std::vector<std::pair<double, int>>>& euclideanDistances) {
    std::vector<int> neighborhoodIndexes(totalPopulationSize * neighborNum);
    for (std::size_t i = 0; i < totalPopulationSize; i++) {
        std::sort(euclideanDistances[i].begin(), euclideanDistances[i].end());
        for (std::size_t j = 0; j < neighborNum; j++) {
            neighborhoodIndexes[i * neighborNum + j] = euclideanDistances[i][j].second;
        }
    }
    return neighborhoodIndexes;
}

template <typename DecisionVariableType>
std::tuple<std::vector<int>, std::vector<int>> MpMoead<DecisionVariableType>::GenerateExternalNeighbourhood(
    std::vector<int>& neighborhoodIndexes, std::vector<int>& populationSizes) {
    std::vector<int> unduplicatedNeighborhoodIndexes;
    std::vector<int> neighborhoodSizes;
    for (int i = 0; i < parallelSize; i++) {
        std::vector<int> indexes(neighborhoodIndexes.begin() + (i == 0 ? 0 : populationSizes[i - 1] * neighborNum),
                                 neighborhoodIndexes.begin() + populationSizes[i] * neighborNum);
        // 重複を削除
        std::sort(indexes.begin(), indexes.end());
        indexes.erase(std::unique(indexes.begin(), indexes.end()), indexes.end());

        // 自分の担当する解のインデックスを削除
        int start = std::reduce(populationSizes.begin(), populationSizes.begin() + i);
        int end = start + populationSizes[i] - 1;
        indexes.erase(
            std::remove_if(indexes.begin(), indexes.end(), [&](int index) { return start <= index && index <= end; }));

        unduplicatedNeighborhoodIndexes.insert(unduplicatedNeighborhoodIndexes.end(), indexes.begin(), indexes.end());
        neighborhoodSizes.push_back(indexes.size());
    }
    return {unduplicatedNeighborhoodIndexes, neighborhoodSizes};
}

template <typename DecisionVariableType>
std::vector<double> MpMoead<DecisionVariableType>::GetWeightVectorMatchingIndex(std::vector<double> weightVectors,
                                                                                std::vector<int> indexes) {
    std::vector<double> matchingWeightVectors;
    for (int i = 0; i < indexes.size(); i++) {
        matchingWeightVectors.insert(matchingWeightVectors.end(), weightVectors.begin() + indexes[i] * objectiveNum,
                                     weightVectors.begin() + (indexes[i] + 1) * objectiveNum);
    }
    return matchingWeightVectors;
}

template <typename DecisionVariableType>
std::pair<std::vector<int>, std::vector<double>> MpMoead<DecisionVariableType>::ScatterExternalNeighborhood(
    std::vector<int>& neighborhoodIndexes, std::vector<int>& neighborhoodSizes, std::vector<double>& weightVectors) {
    int receivedDataCount;
    MPI_Scatter(neighborhoodSizes.data(), 1, MPI_INT, &receivedDataCount, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> dataCounts;
    std::vector<int> displacements;
    if (rank == 0) {
        std::tie(dataCounts, displacements) = GenerateDataCountsAndDisplacements(neighborhoodSizes, 1, parallelSize);
    }
    std::vector<int> receivedNeighborhoodIndexes(receivedDataCount);
    MPI_Scatterv(neighborhoodIndexes.data(), dataCounts.data(), displacements.data(), MPI_INT,
                 receivedNeighborhoodIndexes.data(), receivedDataCount, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<double> receivedWeightVectors(receivedDataCount * objectiveNum);
    if (rank == 0) {
        std::tie(dataCounts, displacements) = GenerateDataCountsAndDisplacements(neighborhoodSizes, objectiveNum, parallelSize);
    }
    MPI_Scatterv(weightVectors.data(), dataCounts.data(), displacements.data(), MPI_DOUBLE, receivedWeightVectors.data(),
                 receivedDataCount * objectiveNum, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return {receivedNeighborhoodIndexes, receivedWeightVectors};
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::InitializePopulation() {
    // 母集団を初期化
    solutions = sampling->Sample(populationSize, decisionVariableNum);
    objectiveSets.reserve(populationSize);
    for (const auto& solution : solutions) {
        objectiveSets.push_back(problem->ComputeObjectiveSet(solution));
    }

    // 近傍を初期化
    int size = externalNeighboringWeightVectors.size();
    allNeighborSolutions = sampling->Sample(size, decisionVariableNum);
    allNeighborObjectiveSets.reserve(size);
    for (const auto& solution : allNeighborSolutions) {
        allNeighborObjectiveSets.push_back(problem->ComputeObjectiveSet(solution));
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::InitializeIdealPoint() {
    Eigen::ArrayXd tmp = objectiveSets[0];
    for (int i = 1; i < populationSize; i++) {
        tmp = tmp.min(objectiveSets[i]);
    }
    for (neighborObjectiveSet : allNeighborObjectiveSets) {
        tmp = tmp.min(neighborObjectiveSet);
    }
    idealPoint = tmp;
}

// template <typename DecisionVariableType>
// Eigen::ArrayX<DecisionVariableType> MpMoead<DecisionVariableType>::GenerateNewSolution(int index) {
//     std::vector<int> childrenIndex = selection->Select(crossover->GetParentNum(), neighborhoodIndexes[index]);
//     std::vector<Eigen::ArrayX<DecisionVariableType>> parents;
//     for (const auto& i : childrenIndex) {
//         parents.push_back(solutions[i]);
//     }
//     Eigen::ArrayX<DecisionVariableType> newSolution = crossover->Cross(parents);
//     mutation->Mutate(newSolution);
//     return newSolution;
// }

// template <typename DecisionVariableType>
// void MpMoead<DecisionVariableType>::UpdateIdealPoint(Eigen::ArrayXd objectiveSet) {
//     idealPoint = idealPoint.min(objectiveSet);
// }

// template <typename DecisionVariableType>
// void MpMoead<DecisionVariableType>::UpdateNeighboringSolutions(int index, Eigen::ArrayX<DecisionVariableType> solution,
//                                                                Eigen::ArrayXd objectiveSet) {
//     for (auto&& i : neighborhoodIndexes[index]) {
//         double newSubObjective = decomposition->ComputeObjective(weightVectors[i], objectiveSet, idealPoint);
//         double oldSubObjective = decomposition->ComputeObjective(weightVectors[i], objectiveSets[i], idealPoint);
//         if (newSubObjective < oldSubObjective) {
//             solutions[i] = solution;
//             objectiveSets[i] = objectiveSet;
//         }
//     }
// }

}  // namespace Eacpp
