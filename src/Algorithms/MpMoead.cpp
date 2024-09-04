#include "Algorithms/MpMoead.h"

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
#include "Utils/TemplateType.h"
#include "Utils/Utils.h"

namespace Eacpp {

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
void MpMoead<DecisionVariableType>::Initialize(int totalPopulationSize, int H) {
    std::vector<double> allWeightVectors;
    std::vector<int> allNeighborhoodIndexes;
    if (rank == 0) {
        allWeightVectors = GenerateAllWeightVectors(H);
        allNeighborhoodIndexes = GenerateAllNeighborhoods(totalPopulationSize, allWeightVectors);
    }

    CaluculatePopulationNum(totalPopulationSize);

    std::vector<int> populationSizes;
    if (rank == 0) {
        populationSizes = GeneratePopulationSizes(totalPopulationSize);
    }

    std::vector<double> receivedWeightVectors = ScatterWeightVector(populationSizes, parallelSize);
    ConvertWeightVectorsToEigenArrayXd(receivedWeightVectors);

    std::vector<int> receivedNeighborhoodIndexes =
        ScatterNeighborhoodIndexes(allNeighborhoodIndexes, populationSizes, parallelSize);
    ConvertNeighborhoodIndexesToVector2d(receivedNeighborhoodIndexes);

    std::vector<double> receivedNeighborWeightVectors =
        SendNeighborWeightVectors(allWeightVectors, allNeighborhoodIndexes, populationSizes, parallelSize);

    ConvertNeighborWeightVectorToEigenArrayXd(receivedNeighborWeightVectors);
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::InitializeIsland() {
    allNeighborSolutions.reserve(populationSize);
    allNeighborObjectiveSets.reserve(populationSize);
    for (int i = 0; i < populationSize; i++) {
        std::vector<Eigen::ArrayX<DecisionVariableType>> neighborSolutions;
        std::vector<Eigen::ArrayXd> neighborObjectiveSets;
        for (int j = 0; j < neighborNum; j++) {
            neighborSolutions.push_back(sampling->Sample(1, decisionVariableNum)[0]);
            neighborObjectiveSets.push_back(problem->ComputeObjectiveSet(neighborSolutions[j]));
        }
        neighborSolutions.push_back(neighborSolutions);
        neighborObjectiveSets.push_back(neighborObjectiveSets);
    }
}

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
void MpMoead<DecisionVariableType>::CaluculatePopulationNum(int totalPopulationSize) {
    populationSize = totalPopulationSize / parallelSize;
    if (rank < totalPopulationSize % parallelSize) {
        populationSize++;
    }
}

template <typename DecisionVariableType>
std::vector<double> MpMoead<DecisionVariableType>::GenerateAllWeightVectors(int H) {
    std::vector<double> takedSetTop(H + 1);
    std::iota(takedSetTop.begin(), takedSetTop.end(), 0);
    std::vector<std::vector<double>> product = Product(takedSetTop, objectiveNum);
    product.erase(std::remove_if(product.begin(), product.end(), [&](auto v) { return std::reduce(v.begin(), v.end()) != H; }),
                  product.end());

    // productを1次元に変換
    std::vector<double> allWeightVectors;
    allWeightVectors.reserve(product.size() * product[0].size());
    for (auto&& v : product) {
        allWeightVectors.insert(allWeightVectors.end(), v.begin(), v.end());
    }

    return allWeightVectors;
}

template <typename DecisionVariableType>
std::vector<int> MpMoead<DecisionVariableType>::GenerateAllNeighborhoods(int totalPopulationSize,
                                                                         std::vector<double>& allWeightVectors) {
    // 各重みベクトル間のユークリッド距離を計算
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

    // 近傍インデックスを計算
    std::vector<int> allNeighborhoodIndexes(totalPopulationSize * neighborNum);
    for (int i = 0; i < totalPopulationSize; i++) {
        std::sort(euclideanDistances[i].begin(), euclideanDistances[i].end());
        for (int j = 0; j < neighborNum; j++) {
            allNeighborhoodIndexes[i * neighborNum + j] = euclideanDistances[i][j].second;
        }
    }

    return allNeighborhoodIndexes;
}

template <typename DecisionVariableType>
std::vector<int> MpMoead<DecisionVariableType>::GeneratePopulationSizes(int totalPopulationSize) {
    std::vector<int> populationSizes(parallelSize);
    for (int i = 0; i < parallelSize; i++) {
        populationSizes[i] = totalPopulationSize / parallelSize;
        if (i < totalPopulationSize % parallelSize) {
            populationSizes[i]++;
        }
    }
    return populationSizes;
}

template <typename DecisionVariableType>
std::vector<double> MpMoead<DecisionVariableType>::ScatterWeightVector(std::vector<double>& allWeightVectors,
                                                                       std::vector<int>& populationSizes) {
    std::vector<int> dataCounts;
    std::vector<int> displacements;
    if (rank == 0) {
        std::tie(dataCounts, displacements) = GenerateDataCountsAndDisplacements(populationSizes, objectiveNum);
    }
    int receivedDataCount = populationSize * objectiveNum;
    std::vector<double> receivedWeightVectors(receivedDataCount);
    MPI_Scatterv(allWeightVectors.data(), dataCounts.data(), displacements.data(), MPI_DOUBLE, receivedWeightVectors.data(),
                 receivedDataCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return receivedWeightVectors;
}

template <typename DecisionVariableType>
std::vector<int> MpMoead<DecisionVariableType>::ScatterNeighborhoodIndexes(std::vector<int>& allNeighborhoodIndexes,
                                                                           std::vector<int>& populationSizes) {
    std::vector<int> dataCounts;
    std::vector<int> displacements;
    if (rank == 0) {
        std::tie(dataCounts, displacements) = GenerateDataCountsAndDisplacements(populationSizes, neighborNum, parallelSize);
    }
    receivedDataCount = populationSize * neighborNum;
    std::vector<int> receivedNeighborhoodIndexes(receivedDataCount);
    MPI_Scatterv(allNeighborhoodIndexes.data(), dataCounts.data(), displacements.data(), MPI_INT,
                 receivedNeighborhoodIndexes.data(), receivedDataCount, MPI_INT, 0, MPI_COMM_WORLD);
    return receivedNeighborhoodIndexes;
}

template <typename DecisionVariableType>
std::vector<double> MpMoead<DecisionVariableType>::SendNeighborWeightVectors(std::vector<double>& allWeightVectors,
                                                                             std::vector<int>& allNeighborhoodIndexes,
                                                                             std::vector<int>& populationSizes) {
    // MARK: MPI_Scatterを使うようにする
    std::vector<int> dataCounts;
    std::vector<int> displacements;
    if (rank == 0) {
        std::tie(dataCounts, displacements) =
            GenerateDataCountsAndDisplacements(populationSizes, neighborNum * (objectiveNum + 1), parallelSize);
    }
    int receivedDataCount = populationSize * neighborNum * (objectiveNum + 1);
    std::vector<double> receivedNeighborWeightVectors(receivedDataCount);

    if (rank == 0) {
        std::vector<MPI_Request> request(parallelSize - 1);
        for (int i = 0; i < parallelSize; i++) {
            // 重みベクトルの重複なしのインデックスを取得
            std::vector<int> weightVectorIndexes(
                allNeighborhoodIndexes.begin() + (i == 0 ? 0 : populationSizes[i - 1] * neighborNum),
                allNeighborhoodIndexes.begin() + populationSizes[i] * neighborNum);
            std::sort(weightVectorIndexes.begin(), weightVectorIndexes.end());
            weightVectorIndexes.erase(std::unique(weightVectorIndexes.begin(), weightVectorIndexes.end()),
                                      weightVectorIndexes.end());

            // インデックスに対応する重みベクトルを取得
            std::vector<double> neighborWeightVectorsToSend;
            neighborWeightVectorsToSend.reserve(dataCounts[i]);
            for (int i = 0; i < weightVectorIndexes.size(); i++) {
                neighborWeightVectorsToSend.push_back(weightVectorIndexes[i]);
                neighborWeightVectorsToSend.insert(neighborWeightVectorsToSend.end(),
                                                   allWeightVectors.begin() + weightVectorIndexes[i] * objectiveNum,
                                                   allWeightVectors.begin() + (weightVectorIndexes[i] + 1) * objectiveNum);
            }

            if (i == 0) {
                std::copy(neighborWeightVectorsToSend.begin(), neighborWeightVectorsToSend.end(),
                          receivedNeighborWeightVectors.begin());
            } else {
                MPI_Isend(neighborWeightVectorsToSend.data(), dataCounts[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &request[i - 1]);
            }
        }
        MPI_Waitall(parallelSize - 1, request.data(), MPI_STATUSES_IGNORE);
    } else {
        MPI_Request request;
        MPI_Irecv(receivedNeighborWeightVectors.data(), receivedDataCount, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }

    return receivedNeighborWeightVectors;
}

template <typename DecisionVariableType>
std::tuple<std::vector<int>, std::vector<int>> MpMoead<DecisionVariableType>::GenerateDataCountsAndDisplacements(
    std::vector<int>& populationSizes, int dataSize) {
    std::vector<int> dataCounts(parallelSize);
    std::vector<int> displacements(parallelSize);
    for (int i = 0; i < parallelSize; i++) {
        dataCounts[i] = populationSizes[i] * dataSize;
        displacements[i] = i == 0 ? 0 : displacements[i - 1] + dataCounts[i - 1];
    }
    return {dataCounts, displacements};
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::ConvertWeightVectorsToEigenArrayXd(std::vector<double>& allWeightVectors) {
    weightVectors.reserve(populationSize);
    for (int i = 0; i < populationSize; i++) {
        weightVectors.push_back(Eigen::Map<Eigen::ArrayXd>(receivedWeightVectors.data() + i * objectiveNum, objectiveNum));
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::ConvertNeighborhoodIndexesToVector2d(std::vector<int>& allNeighborhoods) {
    neighborhoodIndexes.reserve(populationSize);
    for (int i = 0; i < populationSize; i++) {
        neighborhoodIndexes.push_back(std::vector<int>(receivedNeighborhoodIndexes.begin() + i * neighborNum,
                                                       receivedNeighborhoodIndexes.begin() + (i + 1) * neighborNum));
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::ConvertNeighborWeightVectorToEigenArrayXd(
    std::vector<double>& receivedNeighborWeightVectors) {
    int receivedDataCount = receivedNeighborWeightVectors.size() / (objectiveNum + 1);
    neighborWeightVectorIndexes.reserve(receivedDataCount);
    allNeighborWeightVectors.reserve(receivedDataCount);
    for (int i = 0; i < receivedDataCount; i++) {
        neighborWeightVectorIndexes.push_back(receivedNeighborWeightVectors[i * (objectiveNum + 1)]);
        allNeighborWeightVectors.push_back(
            Eigen::Map<Eigen::ArrayXd>(receivedNeighborWeightVectors.data() + i * (objectiveNum + 1) + 1, objectiveNum));
    }
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
    int size = allNeighborWeightVectors.size();
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