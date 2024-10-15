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

#include "Algorithms/IMoead.h"
#include "Algorithms/MoeadInitializer.h"
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

constexpr int maxBufferSize = 100;
constexpr int dataSizeTag = 0;
constexpr int messageTag = 1;

template <typename DecisionVariableType>
class MpMoead : public IMoead<DecisionVariableType> {
   public:
    MpMoead(int generationNum, int neighborhoodSize, int divisionsNumOfWeightVector, int migrationInterval,
            std::shared_ptr<ICrossover<DecisionVariableType>> crossover, std::shared_ptr<IDecomposition> decomposition,
            std::shared_ptr<IMutation<DecisionVariableType>> mutation, std::shared_ptr<IProblem<DecisionVariableType>> problem,
            std::shared_ptr<IRepair<DecisionVariableType>> repair, std::shared_ptr<ISampling<DecisionVariableType>> sampling,
            std::shared_ptr<ISelection> selection)
        : generationNum(generationNum),
          neighborhoodSize(neighborhoodSize),
          migrationInterval(migrationInterval),
          divisionsNumOfWeightVector(divisionsNumOfWeightVector) {
        if (!crossover || !decomposition || !mutation || !problem || !repair || !sampling || !selection) {
            throw std::invalid_argument("Null pointer is passed");
        }

        this->crossover = crossover;
        this->decomposition = decomposition;
        this->mutation = mutation;
        this->problem = problem;
        this->repair = repair;
        this->sampling = sampling;
        this->selection = selection;
        decisionVariablesNum = problem->DecisionVariablesNum();
        objectivesNum = problem->ObjectivesNum();
        currentGeneration = 0;
    }
    virtual ~MpMoead() {}

    void Run() override;
    void Initialize() override;
    void Update() override;
    bool IsEnd() const override;
    std::vector<Eigen::ArrayXd> GetObjectivesList() const override;
    std::vector<Eigen::ArrayX<DecisionVariableType>> GetSolutionList() const override;

    void InitializeMpi();
    void InitializeIsland();

   private:
    int totalPopulationSize;
    int generationNum;
    int currentGeneration;
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
    MoeadInitializer initializer;
    std::vector<int> internalIndexes;
    std::vector<int> externalIndexes;
    std::set<int> updatedSolutionIndexes;
    std::unordered_map<int, Individual<DecisionVariableType>> individuals;
    std::unordered_map<int, Individual<DecisionVariableType>> clonedExternalIndividuals;
    std::vector<int> ranksForExternalIndividuals;
    std::set<int> neighboringRanks;

    std::vector<int> GenerateInternalIndexes();
    std::pair<std::vector<int>, std::vector<int>> GenerateExternalNeighborhood(std::vector<int>& neighborhoodIndexes,
                                                                               std::vector<int>& populationSizes);
    std::vector<double> GetWeightVectorsMatchingIndexes(std::vector<double>& weightVectors, std::vector<int>& indexes);
    void CalculateNeighboringRanks();
    void InitializeIndividualAndWeightVector(std::vector<Eigen::ArrayXd>& weightVectors,
                                             std::vector<std::vector<int>>& neighborhoodIndexes,
                                             std::vector<Eigen::ArrayXd>& externalNeighboringWeightVectors);
    std::vector<std::vector<double>> ScatterPopulation();

    void InitializePopulation();
    void InitializeExternalPopulation(std::vector<std::vector<double>>& receivedSolutions);
    void InitializeObjectives();
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

    while (!IsEnd()) {
        Update();
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::Initialize() {
    totalPopulationSize = initializer.CalculatePopulationSize(divisionsNumOfWeightVector, objectivesNum);
    InitializeMpi();
    InitializeIsland();
    InitializeIdealPoint();
    currentGeneration = 0;
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::Update() {
    MakeLocalCopyOfExternalIndividuals();

    for (auto&& i : internalIndexes) {
        Individual<DecisionVariableType> newIndividual = GenerateNewIndividual(i);
        if (!problem->IsFeasible(newIndividual)) {
            repair->Repair(newIndividual);
        }
        problem->ComputeObjectiveSet(newIndividual);
        decomposition->UpdateIdealPoint(newIndividual.objectives);
        UpdateNeighboringIndividuals(i, newIndividual);
    }

    currentGeneration++;

    if (currentGeneration % migrationInterval == 0) {
        std::vector<std::vector<double>> messages;
        SendMessages();
        messages = ReceiveMessages();

        updatedSolutionIndexes.clear();
        for (auto&& message : messages) {
            UpdateWithMessage(message);
        }
    }
}

template <typename DecisionVariableType>
bool MpMoead<DecisionVariableType>::IsEnd() const {
    return currentGeneration >= generationNum;
}

template <typename DecisionVariableType>
std::vector<Eigen::ArrayXd> MpMoead<DecisionVariableType>::GetObjectivesList() const {
    std::vector<Eigen::ArrayXd> objectives;
    for (auto&& i : internalIndexes) {
        objectives.push_back(individuals.at(i).objectives);
    }

    return objectives;
}

template <typename DecisionVariableType>
std::vector<Eigen::ArrayX<DecisionVariableType>> MpMoead<DecisionVariableType>::GetSolutionList() const {
    std::vector<Eigen::ArrayX<DecisionVariableType>> solutions;
    for (auto&& i : internalIndexes) {
        solutions.push_back(individuals.at(i).solution);
    }

    return solutions;
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

/* FIXME:
 近傍3，ノード数5の場合，C2からC0に解を送信する必要があるが，C2は自分の近傍の解の近傍に自分の解が含まれている場合にC1・C3に送信するため，できていない．
 加えて，ノード数と母集団サイズが近い場合，初期化処理が停止してしまう．
 */
template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::InitializeIsland() {
    std::vector<double> weightVectors1d;
    std::vector<int> neighborhoodIndexes1d;
    if (rank == 0) {
        initializer.GenerateWeightVectorsAndNeighborhoods(divisionsNumOfWeightVector, objectivesNum, neighborhoodSize,
                                                          weightVectors1d, neighborhoodIndexes1d);
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

    externalIndexes = Scatterv(noduplicateNeighborhoodIndexes, neighborhoodSizes, 1, rank, parallelSize);
    std::vector<double> receivedExternalNeighboringWeightVectors =
        Scatterv(sendExternalNeighboringWeightVectors, neighborhoodSizes, objectivesNum, rank, parallelSize);

    std::vector<Eigen::ArrayXd> weightVectors = TransformToEigenArrayX2d(receivedWeightVectors, objectivesNum);
    std::vector<std::vector<int>> neighborhoodIndexes = TransformTo2d(receivedNeighborhoodIndexes, neighborhoodSize);
    std::vector<Eigen::ArrayXd> externalNeighboringWeightVectors =
        TransformToEigenArrayX2d(receivedExternalNeighboringWeightVectors, objectivesNum);

    internalIndexes = GenerateInternalIndexes();
    CalculateNeighboringRanks();
    InitializePopulation();

    auto receivedExternalSolutions = ScatterPopulation();

    InitializeExternalPopulation(receivedExternalSolutions);
    InitializeObjectives();
    InitializeIndividualAndWeightVector(weightVectors, neighborhoodIndexes, externalNeighboringWeightVectors);
}

template <typename DecisionVariableType>
std::vector<int> MpMoead<DecisionVariableType>::GenerateInternalIndexes() {
    int start = CalculateNodeStartIndex(totalPopulationSize, rank, parallelSize);
    int populationSize = CalculateNodeWorkload(totalPopulationSize, rank, parallelSize);
    std::vector<int> solutionIndexes = Rangei(start, start + populationSize - 1, 1);
    return solutionIndexes;
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
void MpMoead<DecisionVariableType>::CalculateNeighboringRanks() {
    ranksForExternalIndividuals.reserve(externalIndexes.size());
    for (auto&& i : externalIndexes) {
        int rank = GetRankFromIndex(totalPopulationSize, i, parallelSize);
        ranksForExternalIndividuals.push_back(rank);
        neighboringRanks.insert(rank);
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::InitializePopulation() {
    int sampleNum = internalIndexes.size();
    std::vector<Individual<DecisionVariableType>> sampledIndividuals = sampling->Sample(sampleNum, decisionVariablesNum);
    for (int i = 0; i < internalIndexes.size(); i++) {
        individuals[internalIndexes[i]] = sampledIndividuals[i];
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::InitializeExternalPopulation(std::vector<std::vector<double>>& receivedSolutions) {
    for (auto&& solutions : receivedSolutions) {
        for (int i = 0; i < solutions.size(); i += decisionVariablesNum + 1) {
            int index = solutions[i];
            if (IsExternal(index)) {
                clonedExternalIndividuals[index].solution =
                    Eigen::Map<Eigen::ArrayXd>(solutions.data() + (i + 1), decisionVariablesNum);
            }
        }
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::InitializeObjectives() {
    for (int i = 0; i < internalIndexes.size(); i++) {
        problem->ComputeObjectiveSet(individuals[internalIndexes[i]]);
    }
    for (int i = 0; i < externalIndexes.size(); i++) {
        problem->ComputeObjectiveSet(clonedExternalIndividuals[externalIndexes[i]]);
    }
}

template <typename DecisionVariableType>
std::vector<std::vector<double>> MpMoead<DecisionVariableType>::ScatterPopulation() {
    std::vector<double> solutionsToSend;
    solutionsToSend.reserve(internalIndexes.size() * (decisionVariablesNum + 1));
    for (auto&& i : internalIndexes) {
        solutionsToSend.push_back(i);
        solutionsToSend.insert(solutionsToSend.end(), individuals[i].solution.begin(), individuals[i].solution.end());
    }
    int dataSize = solutionsToSend.size();

    std::vector<MPI_Request> requests;
    for (auto&& i : neighboringRanks) {
        requests.emplace_back();
        MPI_Isend(&dataSize, 1, MPI_INT, i, dataSizeTag, MPI_COMM_WORLD, &requests.back());
        requests.emplace_back();
        MPI_Isend(solutionsToSend.data(), dataSize, MPI_DOUBLE, i, messageTag, MPI_COMM_WORLD, &requests.back());
    }

    int special = neighborhoodSize / CalculateNodeWorkload(totalPopulationSize, rank, parallelSize);
    if (rank == special) {
        constexpr int dest = 0;
        requests.emplace_back();
        MPI_Isend(&dataSize, 1, MPI_INT, dest, dataSizeTag, MPI_COMM_WORLD, &requests.back());
        requests.emplace_back();
        MPI_Isend(solutionsToSend.data(), dataSize, MPI_DOUBLE, dest, messageTag, MPI_COMM_WORLD, &requests.back());
    } else if (rank == parallelSize - special - 1) {
        int dest = parallelSize - 1;
        requests.emplace_back();
        MPI_Isend(&dataSize, 1, MPI_INT, dest, dataSizeTag, MPI_COMM_WORLD, &requests.back());
        requests.emplace_back();
        MPI_Isend(solutionsToSend.data(), dataSize, MPI_DOUBLE, dest, messageTag, MPI_COMM_WORLD, &requests.back());
    }

    std::vector<std::vector<double>> receivedSolutions;
    for (auto&& rank : neighboringRanks) {
        int receivedDataSize;
        MPI_Recv(&receivedDataSize, 1, MPI_INT, rank, dataSizeTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        receivedSolutions.emplace_back(receivedDataSize);
        MPI_Recv(receivedSolutions.back().data(), receivedDataSize, MPI_DOUBLE, rank, messageTag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }

    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

    return receivedSolutions;
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::InitializeIndividualAndWeightVector(
    std::vector<Eigen::ArrayXd>& weightVectors, std::vector<std::vector<int>>& neighborhoodIndexes,
    std::vector<Eigen::ArrayXd>& externalNeighboringWeightVectors) {
    for (int i = 0; i < internalIndexes.size(); i++) {
        individuals[internalIndexes[i]].weightVector = std::move(weightVectors[i]);
        individuals[internalIndexes[i]].neighborhood = std::move(neighborhoodIndexes[i]);
    }
    for (int i = 0; i < externalIndexes.size(); i++) {
        clonedExternalIndividuals[externalIndexes[i]].weightVector = std::move(externalNeighboringWeightVectors[i]);
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::InitializeIdealPoint() {
    decomposition->InitializeIdealPoint(objectivesNum);

    for (int i = 0; i < internalIndexes.size(); i++) {
        decomposition->UpdateIdealPoint(individuals[internalIndexes[i]].objectives);
    }

    for (int i = 0; i < externalIndexes.size(); i++) {
        decomposition->UpdateIdealPoint(clonedExternalIndividuals[externalIndexes[i]].objectives);
    }
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::MakeLocalCopyOfExternalIndividuals() {
    for (auto&& i : externalIndexes) {
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
    return std::find(internalIndexes.begin(), internalIndexes.end(), index) != internalIndexes.end();
}

template <typename DecisionVariableType>
bool MpMoead<DecisionVariableType>::IsExternal(int index) {
    return std::find(externalIndexes.begin(), externalIndexes.end(), index) != externalIndexes.end();
}

template <typename DecisionVariableType>
std::unordered_map<int, std::vector<double>> MpMoead<DecisionVariableType>::CreateMessages() {
    std::unordered_map<int, std::vector<double>> dataToSend;
    for (int i = 0; i < externalIndexes.size(); i++) {
        int index = externalIndexes[i];
        bool updated = (individuals[index].solution != clonedExternalIndividuals[index].solution).any();
        if (updated) {
            dataToSend[ranksForExternalIndividuals[i]].push_back(index);
            dataToSend[ranksForExternalIndividuals[i]].insert(dataToSend[ranksForExternalIndividuals[i]].end(),
                                                              individuals[index].solution.begin(),
                                                              individuals[index].solution.end());
        }
    }

    std::vector<double> updatedInternalSolutions;
    for (auto&& i : updatedSolutionIndexes) {
        updatedInternalSolutions.push_back(i);
        updatedInternalSolutions.insert(updatedInternalSolutions.end(), individuals[i].solution.begin(),
                                        individuals[i].solution.end());
    }

    for (auto&& [rank, solutions] : dataToSend) {
        solutions.insert(solutions.end(), updatedInternalSolutions.begin(), updatedInternalSolutions.end());
    }

    int special = neighborhoodSize / CalculateNodeWorkload(totalPopulationSize, rank, parallelSize);
    if (rank == special) {
        constexpr int dest = 0;
        dataToSend[dest] = updatedInternalSolutions;
    } else if (rank == parallelSize - special - 1) {
        int dest = parallelSize - 1;
        dataToSend[dest] = updatedInternalSolutions;
    }

    return dataToSend;
}

template <typename DecisionVariableType>
std::vector<int> MpMoead<DecisionVariableType>::GetRanksToReceiveMessages() {
    std::vector<int> ranksToReceiveMessages;
    for (auto&& source : neighboringRanks) {
        int canReceive0;
        int canReceive1;
        MPI_Iprobe(source, dataSizeTag, MPI_COMM_WORLD, &canReceive0, MPI_STATUS_IGNORE);
        MPI_Iprobe(source, messageTag, MPI_COMM_WORLD, &canReceive1, MPI_STATUS_IGNORE);
        if (canReceive0 && canReceive1) {
            ranksToReceiveMessages.push_back(source);
        }
    }

    int special = neighborhoodSize / CalculateNodeWorkload(totalPopulationSize, rank, parallelSize);
    if (rank == 0) {
        int source = special;
        int canReceive0;
        int canReceive1;
        MPI_Iprobe(source, dataSizeTag, MPI_COMM_WORLD, &canReceive0, MPI_STATUS_IGNORE);
        MPI_Iprobe(source, messageTag, MPI_COMM_WORLD, &canReceive1, MPI_STATUS_IGNORE);
        if (canReceive0 && canReceive1) {
            ranksToReceiveMessages.push_back(source);
        }
    } else if (rank == parallelSize - 1) {
        int source = parallelSize - special - 1;
        int canReceive0;
        int canReceive1;
        MPI_Iprobe(source, dataSizeTag, MPI_COMM_WORLD, &canReceive0, MPI_STATUS_IGNORE);
        MPI_Iprobe(source, messageTag, MPI_COMM_WORLD, &canReceive1, MPI_STATUS_IGNORE);
        if (canReceive0 && canReceive1) {
            ranksToReceiveMessages.push_back(source);
        }
    }

    return ranksToReceiveMessages;
}

template <typename DecisionVariableType>
void MpMoead<DecisionVariableType>::SendMessages() {
    // MPI_Isendで使うバッファ
    static std::array<std::vector<int>, maxBufferSize> sendDataSizesBuffers;
    static std::array<std::unordered_map<int, std::vector<double>>, maxBufferSize> sendMessageBuffers;
    static int sendDataSizesBufferIndex = 0;
    static int sendMessageBufferIndex = 0;
    sendDataSizesBufferIndex = (sendDataSizesBufferIndex + 1) % maxBufferSize;
    sendMessageBufferIndex = (sendMessageBufferIndex + 1) % maxBufferSize;

    std::vector<int>& sendDataSizes = sendDataSizesBuffers[sendDataSizesBufferIndex];
    sendDataSizes.clear();
    std::unordered_map<int, std::vector<double>>& sendMessages = sendMessageBuffers[sendMessageBufferIndex];
    sendMessages = CreateMessages();

    // メッセージを送信する
    MPI_Request request;
    for (auto&& [dest, message] : sendMessages) {
        sendDataSizes.push_back(message.size());
        MPI_Isend(&sendDataSizes.back(), 1, MPI_INT, dest, dataSizeTag, MPI_COMM_WORLD, &request);
        MPI_Isend(message.data(), sendDataSizes.back(), MPI_DOUBLE, dest, messageTag, MPI_COMM_WORLD, &request);
    }
}

// bufferが無限バージョン
// template <typename DecisionVariableType>
// void MpMoead<DecisionVariableType>::SendMessages() {
//     static std::vector<int> sendDataSizesBuffers;
//     static std::vector<std::vector<double>> sendMessageBuffers;

//     auto sendMessages = CreateMessages();

//     // メッセージを送信する
//     MPI_Request request;
//     for (auto&& [dest, message] : sendMessages) {
//         sendDataSizesBuffers.push_back(message.size());
//         MPI_Isend(&sendDataSizesBuffers.back(), 1, MPI_INT, dest, dataSizeTag, MPI_COMM_WORLD, &request);
//         sendMessageBuffers.push_back(message);
//         MPI_Isend(sendMessageBuffers.back().data(), sendDataSizesBuffers.back(), MPI_DOUBLE, dest, messageTag,
//         MPI_COMM_WORLD,
//                   &request);
//     }
// }

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
        if (!(IsInternal(index) || IsExternal(index))) {
            continue;
        }

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

}  // namespace Eacpp
