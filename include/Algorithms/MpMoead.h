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
#include "Utils/Utils.h"

namespace Eacpp {

template <typename DecisionVariableType>
class MpMoead {
   private:
    struct Individual {
        Eigen::ArrayX<DecisionVariableType> solution;
        Eigen::ArrayXd objectives;
        std::vector<int> neighborhood;

        Individual() {}

        Individual(std::vector<int> neighborhood) : neighborhood(neighborhood) {}

        bool IsInternal() const { return neighborhood.empty(); }
    };

   public:
    MpMoead(int totalPopulationSize, int generationNum, int decisionVariableNum, int objectiveNum, int neighborNum,
            std::shared_ptr<ICrossover<DecisionVariableType>> crossover, std::shared_ptr<IDecomposition> decomposition,
            std::shared_ptr<IMutation<DecisionVariableType>> mutation, std::shared_ptr<IProblem<DecisionVariableType>> problem,
            std::shared_ptr<ISampling<DecisionVariableType>> sampling, std::shared_ptr<ISelection> selection)
        : totalPopulationSize(totalPopulationSize),
          generationNum(generationNum),
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
    void Initialize(int H);
    void InitializeMpi(int argc, char** argv);
    void InitializeIsland(int H);
    void Update();

   private:
    int totalPopulationSize;
    int generationNum;
    int decisionVariableNum;
    int objectiveNum;
    int neighborNum;
    std::shared_ptr<ICrossover<DecisionVariableType>> crossover;
    std::shared_ptr<IDecomposition> decomposition;
    std::shared_ptr<IMutation<DecisionVariableType>> mutation;
    std::shared_ptr<IProblem<DecisionVariableType>> problem;
    std::shared_ptr<ISampling<DecisionVariableType>> sampling;
    std::shared_ptr<ISelection> selection;
    int rank;
    int parallelSize;
    Eigen::ArrayXd idealPoint;
    std::vector<int> solutionIndexes;
    std::vector<int> externalSolutionIndexes;
    std::unordered_map<int, Individual> individuals;
    std::unordered_map<int, Eigen::ArrayXd> weightVectors;

    // std::vector<Eigen::ArrayXd> weightVectors;
    // std::vector<Eigen::ArrayX<DecisionVariableType>> solutions;
    // std::vector<Eigen::ArrayXd> objectiveSets;
    // std::vector<std::vector<int>> neighborhoodIndexes;

    // std::vector<Eigen::ArrayXd> externalNeighboringWeightVectors;
    // std::vector<Eigen::ArrayXd> externalNeighboringObjectiveSets;
    // std::vector<Eigen::ArrayX<DecisionVariableType>> externalNeighboringSolutions;
    // std::vector<Individual> externalNeighboringSolutionCopies;

    void InitializeIndividualAndWeightVector(std::vector<Eigen::ArrayXd>& weightVectors,
                                             std::vector<std::vector<int>>& neighborhoodIndexes,
                                             std::vector<Eigen::ArrayXd>& externalNeighboringWeightVectors);
    std::vector<int> GenerateSolutionIndexes();
    std::vector<std::vector<double>> GenerateWeightVectors(int H);
    std::vector<int> GenerateNeighborhoods(std::vector<double>& allWeightVectors);
    std::vector<std::vector<std::pair<double, int>>> CalculateEuclideanDistanceBetweenEachWeightVector(
        std::vector<double>& weightVectors);
    std::vector<int> CalculateNeighborhoodIndexes(std::vector<std::vector<std::pair<double, int>>>& euclideanDistances);

    // template <typename T>
    // std::vector<T> Scatter(std::vector<T>& data, std::vector<int>& populationSizes, int singleDataSize);

    std::tuple<std::vector<int>, std::vector<int>> GenerateExternalNeighborhood(std::vector<int>& neighborhoodIndexes,
                                                                                std::vector<int>& populationSizes);
    std::vector<double> GetWeightVectorsMatchingIndexes(std::vector<double> weightVectors, std::vector<int> indexes);
    std::pair<std::vector<int>, std::vector<double>> ScatterExternalNeighborhood(std::vector<int>& neighborhoodIndexes,
                                                                                 std::vector<int>& neighborhoodSizes,
                                                                                 std::vector<double>& weightVectors);

    void InitializePopulation();
    void InitializeIdealPoint();
    Eigen::ArrayX<DecisionVariableType> GenerateNewSolution(int index);
    void RepairSolution(Eigen::ArrayX<DecisionVariableType>& solution);
    void UpdateIdealPoint(Eigen::ArrayXd& objectiveSet);
    void updateSolution(int index, Eigen::ArrayX<DecisionVariableType>& solution, Eigen::ArrayXd& objectiveSet);
    void UpdateNeighboringSolutions(int index, Eigen::ArrayX<DecisionVariableType>& solution, Eigen::ArrayXd& objectiveSet,
                                    std::unordered_map<int, Individual> externalIndividualCopies);

#ifdef _TEST_
   public:
    MpMoead(int totalPopulationSize, int generationNum, int decisionVariableNum, int objectiveNum, int neighborNum)
        : totalPopulationSize(totalPopulationSize),
          generationNum(generationNum),
          decisionVariableNum(decisionVariableNum),
          objectiveNum(objectiveNum),
          neighborNum(neighborNum) {}

    friend class MpMoeadTest;
    friend class MpMoeadMpiTest;
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

//     for (int i = 0; i < generationNum; i++) {
//         Update();
//     }

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
void MpMoead<DecisionVariableType>::Initialize(int H) {
    InitializeMpi(nullptr, nullptr);
    InitializeIsland(H);
    InitializePopulation();
    InitializeIdealPoint();
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::InitializeIsland(int H) {
    std::vector<double> weightVectors1d;
    std::vector<int> neighborhoodIndexes1d;
    if (rank == 0) {
        std::vector<std::vector<double>> weightVectors2d = GenerateWeightVectors(H);
        weightVectors1d = TransformTo1d(weightVectors2d);
        neighborhoodIndexes1d = GenerateNeighborhoods(weightVectors1d);
    }

    std::vector<int> populationSizes;
    if (rank == 0) {
        populationSizes = CalculateNodeWorkloads(totalPopulationSize, parallelSize);
    }

    std::vector<double> receivedWeightVectors = Scatterv(weightVectors1d, populationSizes, objectiveNum, rank, parallelSize);

    std::vector<int> receivedNeighborhoodIndexes =
        Scatterv(neighborhoodIndexes1d, populationSizes, neighborNum, rank, parallelSize);

    std::vector<int> noduplicateNeighborhoodIndexes;
    std::vector<int> neighborhoodSizes;
    std::vector<double> sendExternalNeighboringWeightVectors;
    if (rank == 0) {
        std::tie(noduplicateNeighborhoodIndexes, neighborhoodSizes) =
            GenerateExternalNeighborhood(neighborhoodIndexes1d, populationSizes);
        sendExternalNeighboringWeightVectors = GetWeightVectorsMatchingIndexes(weightVectors1d, noduplicateNeighborhoodIndexes);
    }
    std::vector<double> receivedExternalNeighboringWeightVectors;
    std::tie(externalSolutionIndexes, receivedExternalNeighboringWeightVectors) =
        ScatterExternalNeighborhood(noduplicateNeighborhoodIndexes, neighborhoodSizes, sendExternalNeighboringWeightVectors);

    std::vector<Eigen::ArrayXd> weightVectors = TransformToEigenArrayX2d(receivedWeightVectors, objectiveNum);
    std::vector<std::vector<int>> neighborhoodIndexes = TransformTo2d(receivedNeighborhoodIndexes, neighborNum);
    std::vector<Eigen::ArrayXd> externalNeighboringWeightVectors =
        TransformToEigenArrayX2d(receivedExternalNeighboringWeightVectors, objectiveNum);

    solutionIndexes = GenerateSolutionIndexes();

    InitializeIndividualAndWeightVector(weightVectors, neighborhoodIndexes, externalNeighboringWeightVectors);
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::InitializeIndividualAndWeightVector(
    std::vector<Eigen::ArrayXd>& weightVectors, std::vector<std::vector<int>>& neighborhoodIndexes,
    std::vector<Eigen::ArrayXd>& externalNeighboringWeightVectors) {
    for (int i = 0; i < solutionIndexes.size(); i++) {
        individuals[solutionIndexes[i]] = Individual(neighborhoodIndexes[i]);
        this->weightVectors[solutionIndexes[i]] = weightVectors[i];
    }
    for (int i = 0; i < externalSolutionIndexes.size(); i++) {
        individuals[externalSolutionIndexes[i]] = Individual();
        this->weightVectors[externalSolutionIndexes[i]] = externalNeighboringWeightVectors[i];
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::Update() {
    std::unordered_map<int, Individual> externalIndividualCopies;
    for (auto&& i : externalSolutionIndexes) {
        externalIndividualCopies[i] = individuals[i];
    }

    for (auto&& i : solutionIndexes) {
        Eigen::ArrayX<DecisionVariableType> newSolution = GenerateNewSolution(i);
        RepairSolution(newSolution);
        Eigen::ArrayXd newObjectiveSet = problem->ComputeObjectiveSet(newSolution);
        UpdateIdealPoint(newObjectiveSet);
        UpdateNeighboringSolutions(i, newSolution, newObjectiveSet);
    }
}

template <typename DecisionVariableType>
std::vector<int> MpMoead<DecisionVariableType>::GenerateSolutionIndexes() {
    int start = CalculateNodeStartIndex(totalPopulationSize, rank, parallelSize);
    int populationSize = CalculateNodeWorkload(totalPopulationSize, rank, parallelSize);
    std::vector<int> solutionIndexes = Rangei(start, start + populationSize - 1, 1);
    return solutionIndexes;
}

template <typename DecisionVariableType>
std::vector<std::vector<double>> MpMoead<DecisionVariableType>::GenerateWeightVectors(int H) {
    std::vector<double> numeratorOfWeightVector(H + 1);
    std::iota(numeratorOfWeightVector.begin(), numeratorOfWeightVector.end(), 0);
    std::vector<std::vector<double>> product = Product(numeratorOfWeightVector, objectiveNum);
    product.erase(std::remove_if(product.begin(), product.end(), [&](auto v) { return std::reduce(v.begin(), v.end()) != H; }),
                  product.end());
    for (auto&& p : product) {
        for (auto& elem : p) {
            elem /= H;
        }
    }
    return product;
}

template <typename DecisionVariableType>
std::vector<int> MpMoead<DecisionVariableType>::GenerateNeighborhoods(std::vector<double>& allWeightVectors) {
    std::vector<std::vector<std::pair<double, int>>> euclideanDistances =
        CalculateEuclideanDistanceBetweenEachWeightVector(allWeightVectors);
    std::vector<int> neighborhoodIndexes = CalculateNeighborhoodIndexes(euclideanDistances);
    return neighborhoodIndexes;
}

template <typename DecisionVariableType>
std::vector<std::vector<std::pair<double, int>>>
MpMoead<DecisionVariableType>::CalculateEuclideanDistanceBetweenEachWeightVector(std::vector<double>& weightVectors) {
    std::vector<std::vector<std::pair<double, int>>> euclideanDistances(
        totalPopulationSize, std::vector<std::pair<double, int>>(totalPopulationSize));
    for (int i = 0; i < totalPopulationSize; i++) {
        for (int j = 0; j < totalPopulationSize; j++) {
            std::vector<double> diff(objectiveNum);
            for (int k = 0; k < objectiveNum; k++) {
                diff[k] = weightVectors[i * objectiveNum + k] - weightVectors[j * objectiveNum + k];
            }
            double squaredNorm = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
            euclideanDistances[i][j] = std::make_pair(squaredNorm, j);
        }
    }
    return euclideanDistances;
}

template <typename DecisionVariableType>
std::vector<int> MpMoead<DecisionVariableType>::CalculateNeighborhoodIndexes(
    std::vector<std::vector<std::pair<double, int>>>& euclideanDistances) {
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
std::tuple<std::vector<int>, std::vector<int>> MpMoead<DecisionVariableType>::GenerateExternalNeighborhood(
    std::vector<int>& neighborhoodIndexes, std::vector<int>& populationSizes) {
    std::vector<int> noduplicateNeighborhoodIndexes;
    std::vector<int> neighborhoodSizes;
    for (int i = 0; i < parallelSize; i++) {
        int start = std::reduce(populationSizes.begin(), populationSizes.begin() + i);
        int end = start + populationSizes[i];

        std::vector<int> indexes(neighborhoodIndexes.begin() + (start * neighborNum),
                                 neighborhoodIndexes.begin() + (end * neighborNum));

        // 重複を削除
        std::sort(indexes.begin(), indexes.end());
        indexes.erase(std::unique(indexes.begin(), indexes.end()), indexes.end());

        // 自分の担当する解のインデックスを削除
        std::erase_if(indexes, [&](int index) { return start <= index && index < end; });

        noduplicateNeighborhoodIndexes.insert(noduplicateNeighborhoodIndexes.end(), indexes.begin(), indexes.end());
        neighborhoodSizes.push_back(indexes.size());
    }
    return {noduplicateNeighborhoodIndexes, neighborhoodSizes};
}

template <typename DecisionVariableType>
std::vector<double> MpMoead<DecisionVariableType>::GetWeightVectorsMatchingIndexes(std::vector<double> weightVectors,
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
    std::vector<int> receivedNeighborhoodIndexes;
    receivedNeighborhoodIndexes = Scatterv(neighborhoodIndexes, neighborhoodSizes, 1, rank, parallelSize);

    std::vector<double> receivedWeightVectors;
    receivedWeightVectors = Scatterv(weightVectors, neighborhoodSizes, objectiveNum, rank, parallelSize);

    return {receivedNeighborhoodIndexes, receivedWeightVectors};
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::InitializePopulation() {
    std::vector<Eigen::ArrayX<DecisionVariableType>> solutions = sampling->Sample(individuals.size(), decisionVariableNum);
    for (int i = 0; i < individuals.size(); i++) {
        individuals[i].solution = solutions[i];
        individuals[i].objectives = problem->ComputeObjectiveSet(solutions[i]);
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::InitializeIdealPoint() {
    Eigen::ArrayXd minObjectiveSet = individuals.begin()->second.objectives;
    for (auto&& individual : individuals | std::views::drop(1)) {
        minObjectiveSet = minObjectiveSet.min(individual.second.objectives);
    }
    idealPoint = minObjectiveSet;
}

template <typename DecisionVariableType>
Eigen::ArrayX<DecisionVariableType> MpMoead<DecisionVariableType>::GenerateNewSolution(int index) {
    auto parentCandidates = individuals[index].neighborhood | std::views::filter([index](int i) { return i != index; });
    std::vector<int> parentIndexes = selection->Select(crossover->GetParentNum() - 1, parentCandidates);
    std::vector<Eigen::ArrayX<DecisionVariableType>> parentSolutions;
    parentSolutions.push_back(individuals[index].solution);
    for (auto&& i : parentIndexes) {
        parentSolutions.push_back(individuals[i].solution);
    }
    Eigen::ArrayX<DecisionVariableType> newSolution = crossover->Cross(parentSolutions);
    mutation->Mutate(newSolution);
    return newSolution;
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::RepairSolution(Eigen::ArrayX<DecisionVariableType>& solution) {
    if (!problem->IsFeasible(solution)) {
        solution = sampling->Sample(1, decisionVariableNum)[0];
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::UpdateIdealPoint(Eigen::ArrayXd& objectiveSet) {
    idealPoint = idealPoint.min(objectiveSet);
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::updateSolution(int index, Eigen::ArrayX<DecisionVariableType>& solution,
                                                   Eigen::ArrayXd& objectiveSet) {
    double newSubObjective = decomposition->ComputeObjective(weightVectors[index], objectiveSet, idealPoint);
    double oldSubObjective = decomposition->ComputeObjective(weightVectors[index], individuals[index].objectives, idealPoint);
    if (newSubObjective < oldSubObjective) {
        individuals[index].solution = solution;
        individuals[index].objectives = objectiveSet;
        // TODO: 解が更新されたことをflagで管理する
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::UpdateNeighboringSolutions(int index, Eigen::ArrayX<DecisionVariableType>& solution,
                                                               Eigen::ArrayXd& objectiveSet,
                                                               std::unordered_map<int, Individual> externalIndividualCopies) {
    for (auto&& i : individuals[index].neighborhood) {
        double newSubObjective = decomposition->ComputeObjective(weightVectors[i], objectiveSet, idealPoint);
        if (individuals[i].IsInternal()) {
            double oldSubObjective = decomposition->ComputeObjective(weightVectors[i], individuals[i].objectives, idealPoint);
            if (newSubObjective < oldSubObjective) {
                individuals[i].solution = solution;
                individuals[i].objectives = objectiveSet;
            }
        } else {
            double oldSubObjective =
                decomposition->ComputeObjective(weightVectors[i], externalIndividualCopies[i].objectives, idealPoint);
            if (newSubObjective < oldSubObjective) {
                externalIndividualCopies[i].solution = solution;
                externalIndividualCopies[i].objectives = objectiveSet;
            }
        }
    }
}

}  // namespace Eacpp
