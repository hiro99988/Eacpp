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

        Individual(Eigen::ArrayX<DecisionVariableType> solution, Eigen::ArrayXd objectives, std::vector<int> neighborhood)
            : solution(solution), objectives(objectives), neighborhood(neighborhood) {}

        Individual(const Individual& individual)
            : solution(individual.solution), objectives(individual.objectives), neighborhood(individual.neighborhood) {}

        Individual& operator=(const Individual& individual) {
            solution = individual.solution;
            objectives = individual.objectives;
            neighborhood = individual.neighborhood;
            return *this;
        }

        bool IsInternal() const { return !neighborhood.empty(); }
    };

   public:
    MpMoead(int totalPopulationSize, int generationNum, int decisionVariableNum, int objectiveNum, int neighborNum,
            int migrationInterval, int H, std::shared_ptr<ICrossover<DecisionVariableType>> crossover,
            std::shared_ptr<IDecomposition> decomposition, std::shared_ptr<IMutation<DecisionVariableType>> mutation,
            std::shared_ptr<IProblem<DecisionVariableType>> problem, std::shared_ptr<ISampling<DecisionVariableType>> sampling,
            std::shared_ptr<ISelection> selection)
        : totalPopulationSize(totalPopulationSize),
          generationNum(generationNum),
          decisionVariableNum(decisionVariableNum),
          objectiveNum(objectiveNum),
          neighborNum(neighborNum),
          migrationInterval(migrationInterval),
          H(H),
          crossover(crossover),
          decomposition(decomposition),
          mutation(mutation),
          problem(problem),
          sampling(sampling),
          selection(selection) {}
    virtual ~MpMoead() {}

    void Run();
    void Initialize();
    void InitializeMpi();
    void InitializeIsland();
    void Update();

   private:
    int totalPopulationSize;
    int generationNum;
    int decisionVariableNum;
    int objectiveNum;
    int neighborNum;
    int migrationInterval;
    int H;
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
    std::vector<int> updatedSolutionIndexes;

    // std::vector<Eigen::ArrayXd> weightVectors;
    // std::vector<Eigen::ArrayX<DecisionVariableType>> solutions;
    // std::vector<Eigen::ArrayXd> objectiveSets;
    // std::vector<std::vector<int>> neighborhoodIndexes;

    // std::vector<Eigen::ArrayXd> externalNeighboringWeightVectors;
    // std::vector<Eigen::ArrayXd> externalNeighboringObjectiveSets;
    // std::vector<Eigen::ArrayX<DecisionVariableType>> externalNeighboringSolutions;
    // std::vector<Individual> externalNeighboringSolutionCopies;

    std::vector<int> GenerateSolutionIndexes();
    std::vector<std::vector<double>> GenerateWeightVectors(int H);
    std::vector<int> GenerateNeighborhoods(std::vector<double>& allWeightVectors);
    std::vector<std::vector<std::pair<double, int>>> CalculateEuclideanDistanceBetweenEachWeightVector(
        std::vector<double>& weightVectors);
    std::vector<int> CalculateNeighborhoodIndexes(std::vector<std::vector<std::pair<double, int>>>& euclideanDistances);

    // template <typename T>
    // std::vector<T> Scatter(std::vector<T>& data, std::vector<int>& populationSizes, int singleDataSize);

    std::pair<std::vector<int>, std::vector<int>> GenerateExternalNeighborhood(std::vector<int>& neighborhoodIndexes,
                                                                               std::vector<int>& populationSizes);
    std::vector<double> GetWeightVectorsMatchingIndexes(std::vector<double>& weightVectors, std::vector<int>& indexes);
    std::pair<std::vector<int>, std::vector<double>> ScatterExternalNeighborhood(std::vector<int>& neighborhoodIndexes,
                                                                                 std::vector<int>& neighborhoodSizes,
                                                                                 std::vector<double>& weightVectors);
    void InitializeIndividualAndWeightVector(std::vector<Eigen::ArrayXd>& weightVectors,
                                             std::vector<std::vector<int>>& neighborhoodIndexes,
                                             std::vector<Eigen::ArrayXd>& externalNeighboringWeightVectors);

    void InitializePopulation();
    void InitializeIdealPoint();
    std::vector<Eigen::ArrayX<DecisionVariableType>> SelectParents(int index);
    Eigen::ArrayX<DecisionVariableType> GenerateNewSolution(int index);
    void RepairSolution(Eigen::ArrayX<DecisionVariableType>& solution);
    void UpdateIdealPoint(Eigen::ArrayXd& objectiveSet);
    void UpdateSolution(int index, Eigen::ArrayX<DecisionVariableType>& solution, Eigen::ArrayXd& objectiveSet);
    void UpdateNeighboringSolutions(int index, Eigen::ArrayX<DecisionVariableType>& solution, Eigen::ArrayXd& objectiveSet,
                                    std::unordered_map<int, Individual>& externalIndividualCopies);

    void SendMessage(std::unordered_map<int, Individual> externalIndividualCopies);
    void ReceiveMessage();
    void UpdateWithMessage(std::vector<double>& message);
    std::vector<double> GatherAllObjectives();
    std::vector<Eigen::ArrayXd> GetAllObjectives();

#ifdef _TEST_
   public:
    MpMoead(int totalPopulationSize, int generationNum, int decisionVariableNum, int objectiveNum, int neighborNum,
            int migrationInterval, int H)
        : totalPopulationSize(totalPopulationSize),
          generationNum(generationNum),
          decisionVariableNum(decisionVariableNum),
          objectiveNum(objectiveNum),
          neighborNum(neighborNum),
          migrationInterval(migrationInterval),
          H(H) {}

    friend class MpMoeadTest;
    friend class MpMoeadMpiTest;
#endif
};

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::Run() {
    Initialize();
    int repeat = generationNum / migrationInterval;
    for (int i = 0; i < repeat; i++) {
        Update();
    }

    MPI_Finalize();
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::InitializeMpi() {
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(nullptr, nullptr);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &parallelSize);
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::Initialize() {
    InitializeMpi();
    InitializeIsland();
    InitializePopulation();
    InitializeIdealPoint();
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::InitializeIsland() {
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
void MpMoead<DecisionVariableType>::Update() {
    std::unordered_map<int, Individual> externalIndividualCopies;
    for (auto&& i : externalSolutionIndexes) {
        externalIndividualCopies[i] = individuals[i];
    }

    for (int interval = 0; interval < migrationInterval; interval++) {
        for (auto&& i : solutionIndexes) {
            Eigen::ArrayX<DecisionVariableType> newSolution = GenerateNewSolution(i);
            RepairSolution(newSolution);
            Eigen::ArrayXd newObjectiveSet = problem->ComputeObjectiveSet(newSolution);
            UpdateIdealPoint(newObjectiveSet);
            UpdateSolution(i, newSolution, newObjectiveSet);
            UpdateNeighboringSolutions(i, newSolution, newObjectiveSet, externalIndividualCopies);
        }
    }

    SendMessage(externalIndividualCopies);
    ReceiveMessage();
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
std::pair<std::vector<int>, std::vector<int>> MpMoead<DecisionVariableType>::GenerateExternalNeighborhood(
    std::vector<int>& neighborhoodIndexes, std::vector<int>& populationSizes) {
    std::vector<int> noduplicateNeighborhoodIndexes;
    std::vector<int> neighborhoodSizes;
    for (int i = 0; i < populationSizes.size(); i++) {
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
std::vector<double> MpMoead<DecisionVariableType>::GetWeightVectorsMatchingIndexes(std::vector<double>& weightVectors,
                                                                                   std::vector<int>& indexes) {
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
void MpMoead<DecisionVariableType>::InitializePopulation() {
    std::vector<Eigen::ArrayX<DecisionVariableType>> solutions = sampling->Sample(individuals.size(), decisionVariableNum);
    for (int i = 0; auto&& individual : individuals) {
        individual.second.solution = solutions[i];
        individual.second.objectives = problem->ComputeObjectiveSet(solutions[i]);
        i++;
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
std::vector<Eigen::ArrayX<DecisionVariableType>> MpMoead<DecisionVariableType>::SelectParents(int index) {
    std::vector<int> parentCandidates;
    std::copy_if(individuals[index].neighborhood.begin(), individuals[index].neighborhood.end(),
                 std::back_inserter(parentCandidates), [index](int i) { return i != index; });
    std::vector<int> parentIndexes = selection->Select(crossover->GetParentNum() - 1, std::vector<int>(parentCandidates));
    std::vector<Eigen::ArrayX<DecisionVariableType>> parentSolutions;
    parentSolutions.push_back(individuals[index].solution);
    for (auto&& i : parentIndexes) {
        parentSolutions.push_back(individuals[i].solution);
    }
    return parentSolutions;
}

template <typename DecisionVariableType>
Eigen::ArrayX<DecisionVariableType> MpMoead<DecisionVariableType>::GenerateNewSolution(int index) {
    std::vector<Eigen::ArrayX<DecisionVariableType>> parentSolutions = SelectParents(index);
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
void MpMoead<DecisionVariableType>::UpdateSolution(int index, Eigen::ArrayX<DecisionVariableType>& solution,
                                                   Eigen::ArrayXd& objectiveSet) {
    double newSubObjective = decomposition->ComputeObjective(weightVectors[index], objectiveSet, idealPoint);
    double oldSubObjective = decomposition->ComputeObjective(weightVectors[index], individuals[index].objectives, idealPoint);
    if (newSubObjective < oldSubObjective) {
        individuals[index].solution = solution;
        individuals[index].objectives = objectiveSet;
        updatedSolutionIndexes.push_back(index);
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::UpdateNeighboringSolutions(int index, Eigen::ArrayX<DecisionVariableType>& solution,
                                                               Eigen::ArrayXd& objectiveSet,
                                                               std::unordered_map<int, Individual>& externalIndividualCopies) {
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

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::SendMessage(std::unordered_map<int, Individual> externalIndividualCopies) {
    std::unordered_map<int, std::pair<std::set<int>, std::vector<double>>> send;
    for (auto&& i : externalSolutionIndexes) {
        int process = GetRankFromIndex(totalPopulationSize, i, parallelSize);
        send[process] = std::make_pair(std::set<int>(), std::vector<double>());
    }
    for (auto&& i : externalSolutionIndexes) {
        int process = GetRankFromIndex(totalPopulationSize, i, parallelSize);

        bool isUpdated = (individuals[i].solution != externalIndividualCopies[i].solution).all();
        if (isUpdated) {
            send[process].first.insert(i);
            send[process].second.push_back(i);
            send[process].second.insert(send[process].second.end(), externalIndividualCopies[i].solution.begin(),
                                        externalIndividualCopies[i].solution.end());
        }

        for (auto&& j : updatedSolutionIndexes) {
            bool contains = std::find(individuals[j].neighborhood.begin(), individuals[j].neighborhood.end(), i) !=
                            individuals[j].neighborhood.end();
            if (!contains) {
                continue;
            }
            bool alreadyAdded = send[process].first.find(j) != send[process].first.end();
            if (alreadyAdded) {
                continue;
            }
            send[process].first.insert(j);
            send[process].second.push_back(j);
            send[process].second.insert(send[process].second.end(), individuals[j].solution.begin(),
                                        individuals[j].solution.end());
        }
    }

    for (auto&& [process, value] : send) {
        if (value.second.empty()) {
            continue;
        }

        std::cout << "rank: " << rank << " message: ";
        for (auto&& i : value.second) {
            std::cout << i << " ";
        }
        std::cout << std::endl;

        int sendDataSize = value.second.size();
        MPI_Isend(&sendDataSize, 1, MPI_INT, process, 0, MPI_COMM_WORLD, nullptr);
        MPI_Isend(value.second.data(), sendDataSize, MPI_DOUBLE, process, 1, MPI_COMM_WORLD, nullptr);
        std::cout << "send i send" << std::endl;
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::ReceiveMessage() {
    for (auto&& i : externalSolutionIndexes) {
        int isReceived;
        int source = GetRankFromIndex(totalPopulationSize, i, parallelSize);
        MPI_Iprobe(source, 0, MPI_COMM_WORLD, &isReceived, MPI_STATUS_IGNORE);

        if (!isReceived) {
            continue;
        }

        int receiveDataSize;
        MPI_Recv(&receiveDataSize, 1, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::vector<double> receiveData(receiveDataSize);
        MPI_Recv(receiveData.data(), receiveDataSize, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        UpdateWithMessage(receiveData);
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::UpdateWithMessage(std::vector<double>& message) {
    for (int j = 0; j < message.size(); j += decisionVariableNum + 1) {
        int index = message[j];
        Eigen::ArrayX<DecisionVariableType> newSolution =
            Eigen::Map<Eigen::ArrayX<DecisionVariableType>>(message.data() + j + 1, decisionVariableNum);
        Eigen::ArrayXd newObjectives;
        bool isExternal =
            std::find(externalSolutionIndexes.begin(), externalSolutionIndexes.end(), index) != externalSolutionIndexes.end();
        if (isExternal) {
            newObjectives = problem->ComputeObjectiveSet(newSolution);
            individuals[index].solution = newSolution;
            individuals[index].objectives = newObjectives;
        } else {
            newObjectives = problem->ComputeObjectiveSet(newSolution);
            double newSubObjective = decomposition->ComputeObjective(weightVectors[index], newObjectives, idealPoint);
            double oldSubObjective =
                decomposition->ComputeObjective(weightVectors[index], individuals[index].objectives, idealPoint);
            if (newSubObjective < oldSubObjective) {
                individuals[index].solution = newSolution;
                individuals[index].objectives = newObjectives;
                updatedSolutionIndexes.push_back(index);
            }
        }

        UpdateIdealPoint(newObjectives);
    }
}

template <typename DecisionVariableType>
std::vector<double> MpMoead<DecisionVariableType>::GatherAllObjectives() {
    std::vector<double> sendObjectives;
    for (auto&& i : solutionIndexes) {
        sendObjectives.insert(sendObjectives.end(), individuals[i].objectives.begin(), individuals[i].objectives.end());
    }

    int dataSize = sendObjectives.size();
    std::vector<int> dataCounts;
    if (rank == 0) {
        dataCounts.resize(parallelSize);
    }
    MPI_Gather(&dataSize, 1, MPI_INT, dataCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    int totalDataSize = 0;
    std::vector<int> displacements;
    if (rank == 0) {
        for (int i = 0; i < dataCounts.size(); i++) {
            displacements.push_back(totalDataSize);
            totalDataSize += dataCounts[i];
        }
    }

    std::vector<double> receiveObjectives;
    if (rank == 0) {
        receiveObjectives.resize(totalDataSize);
    }
    MPI_Gatherv(sendObjectives.data(), dataSize, MPI_DOUBLE, receiveObjectives.data(), dataCounts.data(), displacements.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return receiveObjectives;
}

template <typename DecisionVariableType>
std::vector<Eigen::ArrayXd> MpMoead<DecisionVariableType>::GetAllObjectives() {
    std::vector<double> receiveObjectives = GatherAllObjectives();
    return TransformToEigenArrayX2d(receiveObjectives, objectiveNum);
}

}  // namespace Eacpp
