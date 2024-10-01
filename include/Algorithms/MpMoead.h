#pragma once

#include <mpi.h>

#include <algorithm>
#include <array>
#include <eigen3/Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <ranges>
#include <set>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "Crossovers/ICrossover.h"
#include "Decompositions/IDecomposition.h"
#include "Individual/Individual.h"
#include "Mutations/IMutation.h"
#include "Problems/IProblem.h"
#include "Repairs/IRepair.h"
#include "Samplings/ISampling.h"
#include "Selections/ISelection.h"
#include "Utils/MpiUtils.h"
#include "Utils/Utils.h"

namespace Eacpp {

constexpr int maxBufferSize = 1000;
constexpr int dataSizeTag = 0;
constexpr int messageTag = 1;

template <typename DecisionVariableType>
class MpMoead {
   public:
    MpMoead(int totalPopulationSize, int generationNum, int decisionVariablesNum, int objectivesNum, int neighborhoodSize,
            int migrationInterval, int divisionsNumOfWeightVector, std::shared_ptr<ICrossover<DecisionVariableType>> crossover,
            std::shared_ptr<IDecomposition> decomposition, std::shared_ptr<IMutation<DecisionVariableType>> mutation,
            std::shared_ptr<IProblem<DecisionVariableType>> problem, std::shared_ptr<IRepair<DecisionVariableType>> repair,
            std::shared_ptr<ISampling<DecisionVariableType>> sampling, std::shared_ptr<ISelection> selection)
        : totalPopulationSize(totalPopulationSize),
          generationNum(generationNum),
          decisionVariablesNum(decisionVariablesNum),
          objectivesNum(objectivesNum),
          neighborhoodSize(neighborhoodSize),
          migrationInterval(migrationInterval),
          divisionsNumOfWeightVector(divisionsNumOfWeightVector),
          crossover(crossover),
          decomposition(decomposition),
          mutation(mutation),
          problem(problem),
          repair(repair),
          sampling(sampling),
          selection(selection) {}
    virtual ~MpMoead() {}

    void Run();
    void Initialize();
    void InitializeMpi();
    void InitializeIsland();
    void Update();
    void WriteAllObjectives();
    void WriteTransitionOfIdealPoint();

   private:
    int totalPopulationSize;
    int generationNum;
    int decisionVariablesNum;
    int objectivesNum;
    int neighborhoodSize;
    int migrationInterval;
    int divisionsNumOfWeightVector;
    std::shared_ptr<ICrossover<DecisionVariableType>> crossover;
    std::shared_ptr<IDecomposition> decomposition;
    std::shared_ptr<IMutation<DecisionVariableType>> mutation;
    std::shared_ptr<IProblem<DecisionVariableType>> problem;
    std::shared_ptr<IRepair<DecisionVariableType>> repair;
    std::shared_ptr<ISampling<DecisionVariableType>> sampling;
    std::shared_ptr<ISelection> selection;
    int rank;
    int parallelSize;

    std::vector<int> solutionIndexes;
    std::vector<int> externalSolutionIndexes;
    std::unordered_map<int, Individual<DecisionVariableType>> individuals;
    std::set<int> updatedSolutionIndexes;
    std::unordered_map<int, Individual<DecisionVariableType>> clonedExternalIndividuals;
    std::vector<int> ranksForExternalIndividuals;
    std::set<int> neighboringRanks;

    std::vector<Eigen::ArrayXd> transitionOfIdealPoint;

    std::vector<int> GenerateSolutionIndexes();
    std::vector<std::vector<double>> GenerateWeightVectors(int H);
    std::vector<int> GenerateNeighborhoods(std::vector<double>& allWeightVectors);
    std::vector<std::vector<std::pair<double, int>>> CalculateEuclideanDistanceBetweenEachWeightVector(
        std::vector<double>& weightVectors);
    std::vector<int> CalculateNeighborhoodIndexes(std::vector<std::vector<std::pair<double, int>>>& euclideanDistances);

    std::pair<std::vector<int>, std::vector<int>> GenerateExternalNeighborhood(std::vector<int>& neighborhoodIndexes,
                                                                               std::vector<int>& populationSizes);
    std::vector<double> GetWeightVectorsMatchingIndexes(std::vector<double>& weightVectors, std::vector<int>& indexes);
    std::pair<std::vector<int>, std::vector<double>> ScatterExternalNeighborhood(std::vector<int>& neighborhoodIndexes,
                                                                                 std::vector<int>& neighborhoodSizes,
                                                                                 std::vector<double>& weightVectors);
    void InitializeIndividualAndWeightVector(std::vector<Eigen::ArrayXd>& weightVectors,
                                             std::vector<std::vector<int>>& neighborhoodIndexes,
                                             std::vector<Eigen::ArrayXd>& externalNeighboringWeightVectors);
    void CalculateRanksForNeighborhood();

    void InitializePopulation();
    void InitializeIdealPoint();
    void MakeLocalCopyOfExternalIndividuals();
    std::vector<Individual<DecisionVariableType>> SelectParents(int index);
    Individual<DecisionVariableType> GenerateNewIndividual(int index);
    void UpdateNeighboringIndividuals(int index, Individual<DecisionVariableType>& newIndividual);
    bool IsInternal(int index);
    bool IsExternal(int index);

    std::unordered_map<int, std::vector<double>> CreateMessages();
    std::vector<int> GetRanksToReceiveMessages();
    void SendMessages();
    std::vector<std::vector<double>> ReceiveMessages();
    void UpdateWithMessage(std::vector<double>& message);

#ifdef _TEST_
   public:
    MpMoead(int totalPopulationSize, int generationNum, int decisionVariableNum, int objectiveNum, int neighborNum,
            int migrationInterval, int H)
        : totalPopulationSize(totalPopulationSize),
          generationNum(generationNum),
          decisionVariablesNum(decisionVariableNum),
          objectivesNum(objectiveNum),
          neighborhoodSize(neighborNum),
          migrationInterval(migrationInterval),
          divisionsNumOfWeightVector(H) {}

    friend class MpMoeadTest;
    friend class MpMoeadTestM;
#endif
};

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::Run() {
    Initialize();

    transitionOfIdealPoint.push_back(decomposition->IdealPoint());

    int repeat = generationNum / migrationInterval;
    for (int i = 0; i < repeat; i++) {
        Update();
        transitionOfIdealPoint.push_back(decomposition->IdealPoint());
    }
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
    InitializeIdealPoint();
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::InitializeIsland() {
    std::vector<double> weightVectors1d;
    std::vector<int> neighborhoodIndexes1d;
    if (rank == 0) {
        std::vector<std::vector<double>> weightVectors2d = GenerateWeightVectors(divisionsNumOfWeightVector);
        weightVectors1d = TransformTo1d(weightVectors2d);
        neighborhoodIndexes1d = GenerateNeighborhoods(weightVectors1d);
    }

    std::vector<int> populationSizes;
    if (rank == 0) {
        populationSizes = CalculateNodeWorkloads(totalPopulationSize, parallelSize);
    }

    std::vector<double> receivedWeightVectors = Scatterv(weightVectors1d, populationSizes, objectivesNum, rank, parallelSize);

    std::vector<int> receivedNeighborhoodIndexes =
        Scatterv(neighborhoodIndexes1d, populationSizes, neighborhoodSize, rank, parallelSize);

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

    std::vector<Eigen::ArrayXd> weightVectors = TransformToEigenArrayX2d(receivedWeightVectors, objectivesNum);
    std::vector<std::vector<int>> neighborhoodIndexes = TransformTo2d(receivedNeighborhoodIndexes, neighborhoodSize);
    std::vector<Eigen::ArrayXd> externalNeighboringWeightVectors =
        TransformToEigenArrayX2d(receivedExternalNeighboringWeightVectors, objectivesNum);

    solutionIndexes = GenerateSolutionIndexes();
    CalculateRanksForNeighborhood();

    InitializePopulation();

    InitializeIndividualAndWeightVector(weightVectors, neighborhoodIndexes, externalNeighboringWeightVectors);
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::Update() {
    MakeLocalCopyOfExternalIndividuals();

    for (int interval = 0; interval < migrationInterval; interval++) {
        for (auto&& i : solutionIndexes) {
            Individual<DecisionVariableType> newIndividual = GenerateNewIndividual(i);
            if (!problem->IsFeasible(newIndividual)) {
                repair->Repair(newIndividual);
            }
            problem->ComputeObjectiveSet(newIndividual);
            decomposition->UpdateIdealPoint(newIndividual.objectives);
            UpdateNeighboringIndividuals(i, newIndividual);
        }
    }

    SendMessages();
    auto messages = ReceiveMessages();
    updatedSolutionIndexes.clear();
    for (auto&& message : messages) {
        UpdateWithMessage(message);
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
    std::vector<std::vector<double>> product = Product(numeratorOfWeightVector, objectivesNum);
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
            std::vector<double> diff(objectivesNum);
            for (int k = 0; k < objectivesNum; k++) {
                diff[k] = weightVectors[i * objectivesNum + k] - weightVectors[j * objectivesNum + k];
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
    std::vector<int> neighborhoodIndexes(totalPopulationSize * neighborhoodSize);
    for (std::size_t i = 0; i < totalPopulationSize; i++) {
        std::sort(euclideanDistances[i].begin(), euclideanDistances[i].end());
        for (std::size_t j = 0; j < neighborhoodSize; j++) {
            neighborhoodIndexes[i * neighborhoodSize + j] = euclideanDistances[i][j].second;
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

        std::vector<int> indexes(neighborhoodIndexes.begin() + (start * neighborhoodSize),
                                 neighborhoodIndexes.begin() + (end * neighborhoodSize));

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
        matchingWeightVectors.insert(matchingWeightVectors.end(), weightVectors.begin() + indexes[i] * objectivesNum,
                                     weightVectors.begin() + (indexes[i] + 1) * objectivesNum);
    }
    return matchingWeightVectors;
}

template <typename DecisionVariableType>
std::pair<std::vector<int>, std::vector<double>> MpMoead<DecisionVariableType>::ScatterExternalNeighborhood(
    std::vector<int>& neighborhoodIndexes, std::vector<int>& neighborhoodSizes, std::vector<double>& weightVectors) {
    std::vector<int> receivedNeighborhoodIndexes;
    receivedNeighborhoodIndexes = Scatterv(neighborhoodIndexes, neighborhoodSizes, 1, rank, parallelSize);

    std::vector<double> receivedWeightVectors;
    receivedWeightVectors = Scatterv(weightVectors, neighborhoodSizes, objectivesNum, rank, parallelSize);

    return {receivedNeighborhoodIndexes, receivedWeightVectors};
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::InitializePopulation() {
    int sampleNum = solutionIndexes.size() + externalSolutionIndexes.size();
    std::vector<Individual<DecisionVariableType>> sampledIndividuals = sampling->Sample(sampleNum, decisionVariablesNum);

    int sampleIndex = 0;
    for (int i = 0; i < solutionIndexes.size(); i++, sampleIndex++) {
        individuals[solutionIndexes[i]] = sampledIndividuals[sampleIndex];
        problem->ComputeObjectiveSet(individuals[solutionIndexes[i]]);
    }
    for (int i = 0; i < externalSolutionIndexes.size(); i++, sampleIndex++) {
        clonedExternalIndividuals[externalSolutionIndexes[i]] = sampledIndividuals[sampleIndex];
        problem->ComputeObjectiveSet(clonedExternalIndividuals[externalSolutionIndexes[i]]);
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::InitializeIndividualAndWeightVector(
    std::vector<Eigen::ArrayXd>& weightVectors, std::vector<std::vector<int>>& neighborhoodIndexes,
    std::vector<Eigen::ArrayXd>& externalNeighboringWeightVectors) {
    for (int i = 0; i < solutionIndexes.size(); i++) {
        individuals[solutionIndexes[i]].weightVector = weightVectors[i];
        individuals[solutionIndexes[i]].neighborhood = neighborhoodIndexes[i];
    }
    for (int i = 0; i < externalSolutionIndexes.size(); i++) {
        clonedExternalIndividuals[externalSolutionIndexes[i]].weightVector = externalNeighboringWeightVectors[i];
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::CalculateRanksForNeighborhood() {
    for (auto&& i : externalSolutionIndexes) {
        int rank = GetRankFromIndex(totalPopulationSize, i, parallelSize);
        ranksForExternalIndividuals.push_back(rank);
        neighboringRanks.insert(rank);
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::InitializeIdealPoint() {
    for (auto&& individual : individuals) {
        decomposition->UpdateIdealPoint(individual.second.objectives);
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::MakeLocalCopyOfExternalIndividuals() {
    for (auto&& i : externalSolutionIndexes) {
        individuals[i] = clonedExternalIndividuals[i];
    }
}

template <typename DecisionVariableType>
std::vector<Individual<DecisionVariableType>> MpMoead<DecisionVariableType>::SelectParents(int index) {
    std::vector<int> parentCandidates;
    std::copy_if(individuals[index].neighborhood.begin(), individuals[index].neighborhood.end(),
                 std::back_inserter(parentCandidates), [index](int i) { return i != index; });
    std::vector<int> parentIndexes = selection->Select(crossover->GetParentNum() - 1, parentCandidates);
    std::vector<Individual<DecisionVariableType>> parents;
    parents.push_back(individuals[index]);
    for (auto&& i : parentIndexes) {
        parents.push_back(individuals[i]);
    }
    return parents;
}

template <typename DecisionVariableType>
Individual<DecisionVariableType> MpMoead<DecisionVariableType>::GenerateNewIndividual(int index) {
    std::vector<Individual<DecisionVariableType>> parents = SelectParents(index);
    Individual<DecisionVariableType> newIndividual = crossover->Cross(parents);
    mutation->Mutate(newIndividual);
    return newIndividual;
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::UpdateNeighboringIndividuals(int index, Individual<DecisionVariableType>& newIndividual) {
    for (auto&& i : individuals[index].neighborhood) {
        double newSubObjective = decomposition->ComputeObjective(individuals[i].weightVector, newIndividual.objectives);
        double oldSubObjective = decomposition->ComputeObjective(individuals[i].weightVector, individuals[i].objectives);
        if (newSubObjective < oldSubObjective) {
            individuals[i].UpdateFrom(newIndividual);
            if (IsInternal(i)) {
                updatedSolutionIndexes.insert(i);
            }
        }
    }
}

template <typename DecisionVariableType>
bool MpMoead<DecisionVariableType>::IsInternal(int index) {
    return std::find(solutionIndexes.begin(), solutionIndexes.end(), index) != solutionIndexes.end();
}

template <typename DecisionVariableType>
bool MpMoead<DecisionVariableType>::IsExternal(int index) {
    return std::find(externalSolutionIndexes.begin(), externalSolutionIndexes.end(), index) != externalSolutionIndexes.end();
}

template <typename DecisionVariableType>
std::unordered_map<int, std::vector<double>> MpMoead<DecisionVariableType>::CreateMessages() {
    std::unordered_map<int, std::set<int>> indexesToSend;
    std::unordered_map<int, std::vector<double>> dataToSend;
    for (int i = 0; i < externalSolutionIndexes.size(); i++) {
        int index = externalSolutionIndexes[i];
        int rank = ranksForExternalIndividuals[i];

        bool isUpdated = (individuals[index].solution != clonedExternalIndividuals[index].solution).any();
        if (isUpdated) {
            indexesToSend[rank].insert(index);
            dataToSend[rank].push_back(index);
            dataToSend[rank].insert(dataToSend[rank].end(), individuals[index].solution.begin(),
                                    individuals[index].solution.end());
        }

        for (auto&& j : updatedSolutionIndexes) {
            bool notContains = std::find(individuals[j].neighborhood.begin(), individuals[j].neighborhood.end(), index) ==
                               individuals[j].neighborhood.end();
            if (notContains) {
                continue;
            }
            bool alreadyAdded = indexesToSend[rank].find(j) != indexesToSend[rank].end();
            if (alreadyAdded) {
                continue;
            }
            indexesToSend[rank].insert(j);
            dataToSend[rank].push_back(j);
            dataToSend[rank].insert(dataToSend[rank].end(), individuals[j].solution.begin(), individuals[j].solution.end());
        }
    }

    return dataToSend;
}

template <typename DecisionVariableType>
std::vector<int> MpMoead<DecisionVariableType>::GetRanksToReceiveMessages() {
    std::vector<int> ranksToReceiveMessages;
    for (auto&& source : neighboringRanks) {
        int canReceive0;
        int canReceive1;
        MPI_Iprobe(source, 0, MPI_COMM_WORLD, &canReceive0, MPI_STATUS_IGNORE);
        MPI_Iprobe(source, 1, MPI_COMM_WORLD, &canReceive1, MPI_STATUS_IGNORE);
        if (canReceive0 && canReceive1) {
            ranksToReceiveMessages.push_back(source);
        }
    }

    return ranksToReceiveMessages;
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::SendMessages() {
    static std::array<std::vector<int>, maxBufferSize> sendDataSizesBuffers;
    static std::array<std::unordered_map<int, std::vector<double>>, maxBufferSize> sendMessageBuffers;
    static int sendDataSizesBufferIndex = 0;
    static int sendMessageBufferIndex = 0;
    sendDataSizesBufferIndex = (sendDataSizesBufferIndex + 1) % sendDataSizesBuffers.size();
    sendMessageBufferIndex = (sendMessageBufferIndex + 1) % sendMessageBuffers.size();

    std::vector<int>& sendDataSizes = sendDataSizesBuffers[sendDataSizesBufferIndex];
    sendDataSizes.clear();
    std::unordered_map<int, std::vector<double>>& sendMessages = sendMessageBuffers[sendMessageBufferIndex];
    sendMessages = CreateMessages();

    // メッセージを送信する
    std::vector<MPI_Request> sendDataSizeRequests;
    std::vector<MPI_Request> sendMessageRequests;
    for (auto&& [dest, message] : sendMessages) {
        sendDataSizes.push_back(message.size());
        sendDataSizeRequests.push_back(MPI_Request());
        MPI_Isend(&sendDataSizes.back(), 1, MPI_INT, dest, dataSizeTag, MPI_COMM_WORLD, &sendDataSizeRequests.back());

        sendMessageRequests.push_back(MPI_Request());
        MPI_Isend(message.data(), sendDataSizes.back(), MPI_DOUBLE, dest, messageTag, MPI_COMM_WORLD,
                  &sendMessageRequests.back());
    }
}

template <typename DecisionVariableType>
std::vector<std::vector<double>> MpMoead<DecisionVariableType>::ReceiveMessages() {
    auto ranksToReceiveMessages = GetRanksToReceiveMessages();

    std::vector<std::vector<double>> receiveMessages;
    for (auto&& i : ranksToReceiveMessages) {
        int receiveDataSize;
        MPI_Recv(&receiveDataSize, 1, MPI_INT, i, dataSizeTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        receiveMessages.push_back(std::vector<double>(receiveDataSize));
        MPI_Recv(receiveMessages.back().data(), receiveDataSize, MPI_DOUBLE, i, messageTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    return receiveMessages;
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::UpdateWithMessage(std::vector<double>& message) {
    for (int i = 0; i < message.size(); i += decisionVariablesNum + 1) {
        int index = message[i];
        Eigen::ArrayX<DecisionVariableType> newSolution =
            Eigen::Map<Eigen::ArrayX<DecisionVariableType>>(message.data() + i + 1, decisionVariablesNum);
        Individual<DecisionVariableType> newIndividual(newSolution);
        problem->ComputeObjectiveSet(newIndividual);
        if (IsExternal(index)) {
            clonedExternalIndividuals[index].UpdateFrom(newIndividual);
        } else {
            double newSubObjective = decomposition->ComputeObjective(individuals[index].weightVector, newIndividual.objectives);
            double oldSubObjective =
                decomposition->ComputeObjective(individuals[index].weightVector, individuals[index].objectives);
            if (newSubObjective < oldSubObjective) {
                individuals[index].UpdateFrom(newIndividual);
                updatedSolutionIndexes.insert(index);
            }
        }

        decomposition->UpdateIdealPoint(newIndividual.objectives);
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::WriteAllObjectives() {
    if (rank == 0) {
        std::filesystem::create_directories("out/data");
        std::filesystem::create_directories("out/data/mp_moead/");
        std::filesystem::create_directories("out/data/mp_moead/objective");

        for (const auto& entry : std::filesystem::directory_iterator("out/data/mp_moead/objective")) {
            std::filesystem::remove(entry.path());
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    std::string filename = "out/data/mp_moead/objective/objective-" + std::to_string(rank) + ".txt";
    std::ofstream ofs(filename);

    for (auto&& i : solutionIndexes) {
        for (int j = 0; j < individuals[i].objectives.size(); j++) {
            ofs << individuals[i].objectives(j);
            if (j != individuals[i].objectives.size() - 1) {
                ofs << " ";
            }
        }
        ofs << std::endl;
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::WriteTransitionOfIdealPoint() {
    if (rank == 0) {
        std::filesystem::create_directories("out/data");
        std::filesystem::create_directories("out/data/mp_moead/");
        std::filesystem::create_directories("out/data/mp_moead/ideal_point");

        for (const auto& entry : std::filesystem::directory_iterator("out/data/mp_moead/ideal_point")) {
            std::filesystem::remove(entry.path());
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    std::string filename = "out/data/mp_moead/ideal_point/ideal_point-" + std::to_string(rank) + ".txt";
    std::ofstream ofs(filename);
    for (auto&& idealPoint : transitionOfIdealPoint) {
        for (int i = 0; i < idealPoint.size(); i++) {
            ofs << idealPoint[i];
            if (i != idealPoint.size() - 1) {
                ofs << " ";
            }
        }
        ofs << std::endl;
    }
}

}  // namespace Eacpp
