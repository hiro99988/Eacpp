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
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "Algorithms/IMoead.h"
#include "Algorithms/MoeadInitializer.h"
#include "Crossovers/ICrossover.h"
#include "Decompositions/IDecomposition.h"
#include "Individual.h"
#include "Mutations/IMutation.h"
#include "Problems/IProblem.h"
#include "Repairs/IRepair.h"
#include "Samplings/ISampling.h"
#include "Selections/ISelection.h"
#include "Utils/FileUtils.h"
#include "Utils/MpiUtils.h"
#include "Utils/Utils.h"

namespace Eacpp {

template <typename DecisionVariableType>
class MpMoeadIdealTopology : public IMoead<DecisionVariableType> {
   public:
    constexpr static int maxBufferSize = 100;
    constexpr static int messageTag = 0;

   private:
    int generationNum;
    int neighborhoodSize;
    int divisionsNumOfWeightVector;
    int migrationInterval;
    int decisionVariablesNum;
    int objectivesNum;
    int singleMessageSize;
    std::shared_ptr<ICrossover<DecisionVariableType>> crossover;
    std::shared_ptr<IDecomposition> decomposition;
    std::shared_ptr<IMutation<DecisionVariableType>> mutation;
    std::shared_ptr<IProblem<DecisionVariableType>> problem;
    std::shared_ptr<IRepair<DecisionVariableType>> repair;
    std::shared_ptr<ISampling<DecisionVariableType>> sampling;
    std::shared_ptr<ISelection> selection;
    int rank;
    int parallelSize;
    int totalPopulationSize;
    int currentGeneration;
    MoeadInitializer initializer;
    std::vector<int> internalIndexes;
    std::vector<int> externalIndexes;
    std::set<int> updatedSolutionIndexes;
    std::unordered_map<int, Individual<DecisionVariableType>> individuals;
    std::unordered_map<int, Individual<DecisionVariableType>>
        clonedExternalIndividuals;
    std::vector<int> ranksForExternalIndividuals;
    std::set<int> neighboringRanks;
    std::vector<int> ranksToSend;
    bool isIdealPointUpdated;
    std::string idealTopologyFilePath;
    std::vector<int> idealTopologyToSend;
    std::vector<int> idealTopologyToReceive;

   public:
    MpMoeadIdealTopology(
        int generationNum, int neighborhoodSize, int divisionsNumOfWeightVector,
        int migrationInterval, std::string idealTopologyFilePath,
        const std::shared_ptr<ICrossover<DecisionVariableType>>& crossover,
        const std::shared_ptr<IDecomposition>& decomposition,
        const std::shared_ptr<IMutation<DecisionVariableType>>& mutation,
        const std::shared_ptr<IProblem<DecisionVariableType>>& problem,
        const std::shared_ptr<IRepair<DecisionVariableType>>& repair,
        const std::shared_ptr<ISampling<DecisionVariableType>>& sampling,
        const std::shared_ptr<ISelection>& selection)
        : generationNum(generationNum),
          neighborhoodSize(neighborhoodSize),
          migrationInterval(migrationInterval),
          divisionsNumOfWeightVector(divisionsNumOfWeightVector),
          idealTopologyFilePath(idealTopologyFilePath) {
        if (!crossover || !decomposition || !mutation || !problem || !repair ||
            !sampling || !selection) {
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
        singleMessageSize = decisionVariablesNum + objectivesNum + 1;
        isIdealPointUpdated = false;
    }
    virtual ~MpMoeadIdealTopology() {}

    int CurrentGeneration() const override {
        return currentGeneration;
    }
    void Run() override;
    void Initialize() override;
    void Update() override;
    bool IsEnd() const override;
    std::vector<Eigen::ArrayXd> GetObjectivesList() const override;
    std::vector<Eigen::ArrayX<DecisionVariableType>> GetSolutionList()
        const override;

    void InitializeMpi();
    void InitializeIsland();

   private:
    void Clear();
    std::pair<std::vector<int>, std::vector<int>> GenerateExternalNeighborhood(
        std::vector<int>& neighborhoodIndexes,
        std::vector<int>& populationSizes);
    std::vector<double> GetWeightVectorsMatchingIndexes(
        std::vector<double>& weightVectors, std::vector<int>& indexes);
    void CalculateNeighboringRanks();
    void InitializeIndividualAndWeightVector(
        std::vector<Eigen::ArrayXd>& weightVectors,
        std::vector<std::vector<int>>& neighborhoodIndexes,
        std::vector<Eigen::ArrayXd>& externalNeighboringWeightVectors);
    void CalculateRanksToSent(const std::vector<int>& neighborhoodIndexes,
                              const std::vector<int>& populationSizes,
                              std::vector<int>& outRanksToSentByRank,
                              std::vector<int>& outSizes);
    std::vector<std::vector<double>> ScatterPopulation();
    void InitializeIdealTopology();

    void InitializePopulation();
    void InitializeExternalPopulation(
        std::vector<std::vector<double>>& receivedSolutions);
    void MakeLocalCopyOfExternalIndividuals();
    std::vector<Individual<DecisionVariableType>> SelectParents(int index);
    Individual<DecisionVariableType> GenerateNewIndividual(int index);
    void UpdateNeighboringIndividuals(
        int index, Individual<DecisionVariableType>& newIndividual);
    bool IsInternal(int index);
    bool IsExternal(int index);
    void UpdateIdealPoint(const Eigen::ArrayXd& objectives);
    void UpdateIdealPointWithMessage(const std::vector<double>& message);

    std::unordered_map<int, std::vector<double>> CreateMessages();
    void SendMessages();
    std::vector<std::vector<double>> ReceiveMessages();
    void UpdateWithMessage(std::vector<double>& message);

#ifdef _TEST_
   public:
    MpMoeadIdealTopology(int totalPopulationSize, int generationNum,
                         int decisionVariableNum, int objectiveNum,
                         int neighborNum, int migrationInterval, int H)
        : totalPopulationSize(totalPopulationSize),
          generationNum(generationNum),
          decisionVariablesNum(decisionVariableNum),
          objectivesNum(objectiveNum),
          neighborhoodSize(neighborNum),
          migrationInterval(migrationInterval),
          divisionsNumOfWeightVector(H) {}

    friend class MpMoeadIdealTopologyTest;
    friend class MpMoeadIdealTopologyTestM;
#endif
};

template <typename DecisionVariableType>
void MpMoeadIdealTopology<DecisionVariableType>::Run() {
    Initialize();

    while (!IsEnd()) {
        Update();
    }
}

template <typename DecisionVariableType>
void MpMoeadIdealTopology<DecisionVariableType>::Initialize() {
    Clear();
    InitializeMpi();
    totalPopulationSize = initializer.CalculatePopulationSize(
        divisionsNumOfWeightVector, objectivesNum);
    decomposition->InitializeIdealPoint(objectivesNum);
    InitializeIsland();
    InitializeIdealTopology();
    currentGeneration = 0;
}

template <typename DecisionVariableType>
void MpMoeadIdealTopology<DecisionVariableType>::Update() {
    if (currentGeneration % migrationInterval == 0) {
        MakeLocalCopyOfExternalIndividuals();
    }

    for (auto&& i : internalIndexes) {
        Individual<DecisionVariableType> newIndividual =
            GenerateNewIndividual(i);
        repair->Repair(newIndividual);
        problem->ComputeObjectiveSet(newIndividual);
        UpdateIdealPoint(newIndividual.objectives);
        UpdateNeighboringIndividuals(i, newIndividual);
    }

    currentGeneration++;

    if (currentGeneration % migrationInterval == 0) {
        SendMessages();
        auto messages = ReceiveMessages();

        updatedSolutionIndexes.clear();
        isIdealPointUpdated = false;
        for (auto&& message : messages) {
            if (message.empty()) {
                continue;
            }

            if (message.size() == objectivesNum) {
                UpdateIdealPointWithMessage(message);
            } else {
                UpdateWithMessage(message);
            }
        }
    }
}

template <typename DecisionVariableType>
bool MpMoeadIdealTopology<DecisionVariableType>::IsEnd() const {
    return currentGeneration >= generationNum;
}

template <typename DecisionVariableType>
std::vector<Eigen::ArrayXd>
MpMoeadIdealTopology<DecisionVariableType>::GetObjectivesList() const {
    std::vector<Eigen::ArrayXd> objectives;
    for (auto&& i : internalIndexes) {
        objectives.push_back(individuals.at(i).objectives);
    }

    return objectives;
}

template <typename DecisionVariableType>
std::vector<Eigen::ArrayX<DecisionVariableType>>
MpMoeadIdealTopology<DecisionVariableType>::GetSolutionList() const {
    std::vector<Eigen::ArrayX<DecisionVariableType>> solutions;
    for (auto&& i : internalIndexes) {
        solutions.push_back(individuals.at(i).solution);
    }

    return solutions;
}

template <typename DecisionVariableType>
void MpMoeadIdealTopology<DecisionVariableType>::Clear() {
    internalIndexes.clear();
    externalIndexes.clear();
    updatedSolutionIndexes.clear();
    individuals.clear();
    clonedExternalIndividuals.clear();
    ranksForExternalIndividuals.clear();
    neighboringRanks.clear();
    ranksToSend.clear();
}

template <typename DecisionVariableType>
void MpMoeadIdealTopology<DecisionVariableType>::InitializeMpi() {
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(nullptr, nullptr);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &parallelSize);
}

template <typename DecisionVariableType>
void MpMoeadIdealTopology<DecisionVariableType>::InitializeIsland() {
    std::vector<double> weightVectors1d;
    std::vector<int> neighborhoodIndexes1d;
    std::vector<int> populationSizes;
    std::vector<int> noduplicateNeighborhoodIndexes;
    std::vector<int> neighborhoodSizes;
    std::vector<double> sendExternalNeighboringWeightVectors;
    std::vector<int> sendRanksToSentByRank;
    std::vector<int> ranksToSentByRankSizes;
    if (rank == 0) {
        initializer.GenerateWeightVectorsAndNeighborhoods(
            divisionsNumOfWeightVector, objectivesNum, neighborhoodSize,
            weightVectors1d, neighborhoodIndexes1d);
        populationSizes =
            CalculateNodeWorkloads(totalPopulationSize, parallelSize);
    }

    std::vector<double> receivedWeightVectors = Scatterv(
        weightVectors1d, populationSizes, objectivesNum, rank, parallelSize);
    std::vector<int> receivedNeighborhoodIndexes =
        Scatterv(neighborhoodIndexes1d, populationSizes, neighborhoodSize, rank,
                 parallelSize);

    if (rank == 0) {
        std::tie(noduplicateNeighborhoodIndexes, neighborhoodSizes) =
            GenerateExternalNeighborhood(neighborhoodIndexes1d,
                                         populationSizes);
        sendExternalNeighboringWeightVectors = GetWeightVectorsMatchingIndexes(
            weightVectors1d, noduplicateNeighborhoodIndexes);
    }

    externalIndexes = Scatterv(noduplicateNeighborhoodIndexes,
                               neighborhoodSizes, 1, rank, parallelSize);
    std::vector<double> receivedExternalNeighboringWeightVectors =
        Scatterv(sendExternalNeighboringWeightVectors, neighborhoodSizes,
                 objectivesNum, rank, parallelSize);

    if (rank == 0) {
        CalculateRanksToSent(neighborhoodIndexes1d, populationSizes,
                             sendRanksToSentByRank, ranksToSentByRankSizes);
    }
    ranksToSend = Scatterv(sendRanksToSentByRank, ranksToSentByRankSizes, 1,
                           rank, parallelSize);

    std::vector<Eigen::ArrayXd> weightVectors =
        TransformToEigenArrayX2d(receivedWeightVectors, objectivesNum);
    std::vector<std::vector<int>> neighborhoodIndexes =
        TransformTo2d(receivedNeighborhoodIndexes, neighborhoodSize);
    std::vector<Eigen::ArrayXd> externalNeighboringWeightVectors =
        TransformToEigenArrayX2d(receivedExternalNeighboringWeightVectors,
                                 objectivesNum);

    internalIndexes =
        GenerateNodeIndexes(totalPopulationSize, rank, parallelSize);
    CalculateNeighboringRanks();
    InitializePopulation();

    auto receivedExternalSolutions = ScatterPopulation();

    InitializeExternalPopulation(receivedExternalSolutions);
    InitializeIndividualAndWeightVector(weightVectors, neighborhoodIndexes,
                                        externalNeighboringWeightVectors);
}

template <typename DecisionVariableType>
std::pair<std::vector<int>, std::vector<int>>
MpMoeadIdealTopology<DecisionVariableType>::GenerateExternalNeighborhood(
    std::vector<int>& neighborhoodIndexes, std::vector<int>& populationSizes) {
    std::vector<int> noduplicateNeighborhoodIndexes;
    std::vector<int> neighborhoodSizes;
    for (int i = 0; i < populationSizes.size(); i++) {
        int start =
            std::reduce(populationSizes.begin(), populationSizes.begin() + i);
        int end = start + populationSizes[i];

        std::vector<int> indexes(
            neighborhoodIndexes.begin() + (start * neighborhoodSize),
            neighborhoodIndexes.begin() + (end * neighborhoodSize));

        // 重複を削除
        std::sort(indexes.begin(), indexes.end());
        indexes.erase(std::unique(indexes.begin(), indexes.end()),
                      indexes.end());

        // 自分の担当する解のインデックスを削除
        std::erase_if(indexes,
                      [&](int index) { return start <= index && index < end; });

        noduplicateNeighborhoodIndexes.insert(
            noduplicateNeighborhoodIndexes.end(), indexes.begin(),
            indexes.end());
        neighborhoodSizes.push_back(indexes.size());
    }
    return {noduplicateNeighborhoodIndexes, neighborhoodSizes};
}

template <typename DecisionVariableType>
std::vector<double>
MpMoeadIdealTopology<DecisionVariableType>::GetWeightVectorsMatchingIndexes(
    std::vector<double>& weightVectors, std::vector<int>& indexes) {
    std::vector<double> matchingWeightVectors;
    for (int i = 0; i < indexes.size(); i++) {
        matchingWeightVectors.insert(
            matchingWeightVectors.end(),
            weightVectors.begin() + indexes[i] * objectivesNum,
            weightVectors.begin() + (indexes[i] + 1) * objectivesNum);
    }
    return matchingWeightVectors;
}

template <typename DecisionVariableType>
void MpMoeadIdealTopology<DecisionVariableType>::CalculateNeighboringRanks() {
    ranksForExternalIndividuals.reserve(externalIndexes.size());
    for (auto&& i : externalIndexes) {
        int rank = GetRankFromIndex(totalPopulationSize, i, parallelSize);
        ranksForExternalIndividuals.push_back(rank);
        neighboringRanks.insert(rank);
    }
}

template <typename DecisionVariableType>
void MpMoeadIdealTopology<DecisionVariableType>::InitializePopulation() {
    int sampleNum = internalIndexes.size();
    std::vector<Individual<DecisionVariableType>> sampledIndividuals =
        sampling->Sample(sampleNum, decisionVariablesNum);
    for (int i = 0; i < internalIndexes.size(); i++) {
        individuals[internalIndexes[i]] = sampledIndividuals[i];
        problem->ComputeObjectiveSet(individuals[internalIndexes[i]]);
        decomposition->UpdateIdealPoint(
            individuals[internalIndexes[i]].objectives);
    }
}

template <typename DecisionVariableType>
void MpMoeadIdealTopology<DecisionVariableType>::InitializeExternalPopulation(
    std::vector<std::vector<double>>& receivedIndividuals) {
    for (auto&& receive : receivedIndividuals) {
        int limit = receive.size() - objectivesNum;
        for (int i = 0; i < limit; i += singleMessageSize) {
            int index = receive[i];
            if (!IsExternal(index)) {
                continue;
            }

            clonedExternalIndividuals[index].solution =
                Eigen::Map<Eigen::ArrayXd>(receive.data() + (i + 1),
                                           decisionVariablesNum);
            clonedExternalIndividuals[index].objectives =
                Eigen::Map<Eigen::ArrayXd>(
                    receive.data() + (i + 1 + decisionVariablesNum),
                    objectivesNum);
            UpdateIdealPoint(clonedExternalIndividuals[index].objectives);
        }

        UpdateIdealPointWithMessage(receive);
    }
}

template <typename DecisionVariableType>
void MpMoeadIdealTopology<DecisionVariableType>::CalculateRanksToSent(
    const std::vector<int>& neighborhoodIndexes,
    const std::vector<int>& populationSizes,
    std::vector<int>& outRanksToSentByRank, std::vector<int>& outSizes) {
    std::vector<std::set<int>> ranksToSentByRank(parallelSize, std::set<int>());
    for (int dest = 0, count = 0; dest < parallelSize;
         count += populationSizes[dest] * neighborhoodSize, ++dest) {
        std::vector<int> neighborhood;
        std::copy(neighborhoodIndexes.begin() + count,
                  neighborhoodIndexes.begin() + count +
                      populationSizes[dest] * neighborhoodSize,
                  std::back_inserter(neighborhood));
        std::sort(neighborhood.begin(), neighborhood.end());
        neighborhood.erase(
            std::unique(neighborhood.begin(), neighborhood.end()),
            neighborhood.end());

        for (auto&& i : neighborhood) {
            int source = GetRankFromIndex(totalPopulationSize, i, parallelSize);
            if (source != dest) {
                ranksToSentByRank[source].insert(dest);
            }
        }
    }

    for (auto&& i : ranksToSentByRank) {
        outSizes.push_back(i.size());
    }

    outRanksToSentByRank.reserve(parallelSize * neighborhoodSize);
    for (auto&& i : ranksToSentByRank) {
        for (auto&& j : i) {
            outRanksToSentByRank.push_back(j);
        }
    }
}

template <typename DecisionVariableType>
std::vector<std::vector<double>>
MpMoeadIdealTopology<DecisionVariableType>::ScatterPopulation() {
    std::vector<double> individualsToSend;
    individualsToSend.reserve(internalIndexes.size() * singleMessageSize +
                              objectivesNum);
    for (auto&& i : internalIndexes) {
        individualsToSend.push_back(i);
        individualsToSend.insert(individualsToSend.end(),
                                 individuals[i].solution.begin(),
                                 individuals[i].solution.end());
        individualsToSend.insert(individualsToSend.end(),
                                 individuals[i].objectives.begin(),
                                 individuals[i].objectives.end());
    }

    individualsToSend.insert(individualsToSend.end(),
                             decomposition->IdealPoint().begin(),
                             decomposition->IdealPoint().end());

    std::vector<MPI_Request> requests;
    requests.reserve(ranksToSend.size());
    for (auto&& i : ranksToSend) {
        requests.emplace_back();
        MPI_Isend(individualsToSend.data(), individualsToSend.size(),
                  MPI_DOUBLE, i, messageTag, MPI_COMM_WORLD, &requests.back());
    }

    std::vector<std::vector<double>> receivedIndividuals;
    receivedIndividuals.reserve(neighboringRanks.size());
    for (auto&& rank : neighboringRanks) {
        MPI_Status status;
        MPI_Probe(rank, messageTag, MPI_COMM_WORLD, &status);
        int receivedDataSize;
        MPI_Get_count(&status, MPI_DOUBLE, &receivedDataSize);
        receivedIndividuals.emplace_back(receivedDataSize);
        MPI_Recv(receivedIndividuals.back().data(), receivedDataSize,
                 MPI_DOUBLE, rank, messageTag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }

    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

    return receivedIndividuals;
}

template <typename DecisionVariableType>
void MpMoeadIdealTopology<DecisionVariableType>::
    InitializeIndividualAndWeightVector(
        std::vector<Eigen::ArrayXd>& weightVectors,
        std::vector<std::vector<int>>& neighborhoodIndexes,
        std::vector<Eigen::ArrayXd>& externalNeighboringWeightVectors) {
    for (int i = 0; i < internalIndexes.size(); i++) {
        individuals[internalIndexes[i]].weightVector =
            std::move(weightVectors[i]);
        individuals[internalIndexes[i]].neighborhood =
            std::move(neighborhoodIndexes[i]);
    }
    for (int i = 0; i < externalIndexes.size(); i++) {
        clonedExternalIndividuals[externalIndexes[i]].weightVector =
            std::move(externalNeighboringWeightVectors[i]);
    }
}

template <typename DecisionVariableType>
void MpMoeadIdealTopology<DecisionVariableType>::InitializeIdealTopology() {
    std::ifstream ifs = OpenInputFile(idealTopologyFilePath);
    std::string line;
    for (int i = 0; i <= rank; ++i) {
        if (!std::getline(ifs, line)) {
            throw std::runtime_error("Not enough lines in idealTopologyFile");
        }
    }
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) {
        int rank = std::stoi(item);
        idealTopologyToSend.push_back(rank);
        if (neighboringRanks.find(rank) == neighboringRanks.end()) {
            idealTopologyToReceive.push_back(rank);
        }
    }
}

template <typename DecisionVariableType>
void MpMoeadIdealTopology<
    DecisionVariableType>::MakeLocalCopyOfExternalIndividuals() {
    for (auto&& i : externalIndexes) {
        individuals[i] = clonedExternalIndividuals[i];
    }
}

template <typename DecisionVariableType>
std::vector<Individual<DecisionVariableType>>
MpMoeadIdealTopology<DecisionVariableType>::SelectParents(int index) {
    std::vector<int> parentCandidates;
    std::copy_if(individuals[index].neighborhood.begin(),
                 individuals[index].neighborhood.end(),
                 std::back_inserter(parentCandidates),
                 [index](int i) { return i != index; });
    std::vector<int> parentIndexes =
        selection->Select(crossover->GetParentNum() - 1, parentCandidates);
    std::vector<Individual<DecisionVariableType>> parents;
    parents.push_back(individuals[index]);
    for (auto&& i : parentIndexes) {
        parents.push_back(individuals[i]);
    }
    return parents;
}

template <typename DecisionVariableType>
Individual<DecisionVariableType>
MpMoeadIdealTopology<DecisionVariableType>::GenerateNewIndividual(int index) {
    std::vector<Individual<DecisionVariableType>> parents =
        SelectParents(index);
    Individual<DecisionVariableType> newIndividual = crossover->Cross(parents);
    mutation->Mutate(newIndividual);
    return newIndividual;
}

template <typename DecisionVariableType>
void MpMoeadIdealTopology<DecisionVariableType>::UpdateNeighboringIndividuals(
    int index, Individual<DecisionVariableType>& newIndividual) {
    for (auto&& i : individuals[index].neighborhood) {
        double newSubObjective = decomposition->ComputeObjective(
            individuals[i].weightVector, newIndividual.objectives);
        double oldSubObjective = decomposition->ComputeObjective(
            individuals[i].weightVector, individuals[i].objectives);
        if (newSubObjective < oldSubObjective) {
            individuals[i].UpdateFrom(newIndividual);
            if (IsInternal(i)) {
                updatedSolutionIndexes.insert(i);
            }
        }
    }
}

template <typename DecisionVariableType>
bool MpMoeadIdealTopology<DecisionVariableType>::IsInternal(int index) {
    return std::find(internalIndexes.begin(), internalIndexes.end(), index) !=
           internalIndexes.end();
}

template <typename DecisionVariableType>
bool MpMoeadIdealTopology<DecisionVariableType>::IsExternal(int index) {
    return std::find(externalIndexes.begin(), externalIndexes.end(), index) !=
           externalIndexes.end();
}

template <typename DecisionVariableType>
void MpMoeadIdealTopology<DecisionVariableType>::UpdateIdealPoint(
    const Eigen::ArrayXd& objectives) {
    auto idealPoint = decomposition->IdealPoint();
    for (int i = 0; i < objectivesNum; i++) {
        if (objectives(i) < idealPoint(i)) {
            isIdealPointUpdated = true;
            decomposition->UpdateIdealPoint(objectives);
            break;
        }
    }
}

template <typename DecisionVariableType>
void MpMoeadIdealTopology<DecisionVariableType>::UpdateIdealPointWithMessage(
    const std::vector<double>& message) {
    if (message.size() % singleMessageSize == objectivesNum) {
        Eigen::ArrayXd receivedIdealPoint = Eigen::Map<const Eigen::ArrayXd>(
            message.data() + (message.size() - objectivesNum), objectivesNum);
        UpdateIdealPoint(receivedIdealPoint);
    }
}

template <typename DecisionVariableType>
std::unordered_map<int, std::vector<double>>
MpMoeadIdealTopology<DecisionVariableType>::CreateMessages() {
    std::vector<double> updatedInternalIndividuals;
    updatedInternalIndividuals.reserve(updatedSolutionIndexes.size() *
                                       singleMessageSize);
    for (auto&& i : updatedSolutionIndexes) {
        updatedInternalIndividuals.push_back(i);
        updatedInternalIndividuals.insert(updatedInternalIndividuals.end(),
                                          individuals[i].solution.begin(),
                                          individuals[i].solution.end());
        updatedInternalIndividuals.insert(updatedInternalIndividuals.end(),
                                          individuals[i].objectives.begin(),
                                          individuals[i].objectives.end());
    }

    std::unordered_map<int, std::vector<double>> dataToSend;
    for (int i = 0; i < externalIndexes.size(); i++) {
        int index = externalIndexes[i];
        bool updated = (individuals[index].solution !=
                        clonedExternalIndividuals[index].solution)
                           .any();
        if (updated) {
            int rank = ranksForExternalIndividuals[i];
            dataToSend[rank].push_back(index);
            dataToSend[rank].insert(dataToSend[rank].end(),
                                    individuals[index].solution.begin(),
                                    individuals[index].solution.end());
            dataToSend[rank].insert(dataToSend[rank].end(),
                                    individuals[index].objectives.begin(),
                                    individuals[index].objectives.end());
        }
    }

    for (auto&& rank : neighboringRanks) {
        dataToSend[rank].insert(dataToSend[rank].end(),
                                updatedInternalIndividuals.begin(),
                                updatedInternalIndividuals.end());
    }

    if (isIdealPointUpdated) {
        for (auto&& rank : idealTopologyToSend) {
            dataToSend[rank].insert(dataToSend[rank].end(),
                                    decomposition->IdealPoint().begin(),
                                    decomposition->IdealPoint().end());
        }
    }

    return dataToSend;
}

template <typename DecisionVariableType>
void MpMoeadIdealTopology<DecisionVariableType>::SendMessages() {
    // MPI_Isendで使うバッファ
    static std::array<std::unordered_map<int, std::vector<double>>,
                      maxBufferSize>
        sendMessageBuffers;
    static int sendMessageBufferIndex = 0;
    sendMessageBufferIndex = (sendMessageBufferIndex + 1) % maxBufferSize;

    std::unordered_map<int, std::vector<double>>& sendMessages =
        sendMessageBuffers[sendMessageBufferIndex];
    sendMessages = CreateMessages();

    // メッセージを送信する
    MPI_Request request;
    for (auto&& [dest, message] : sendMessages) {
        MPI_Isend(message.data(), message.size(), MPI_DOUBLE, dest, messageTag,
                  MPI_COMM_WORLD, &request);
    }
}

// bufferが無限バージョン
// template <typename DecisionVariableType>
// void MpMoeadIdealTopology<DecisionVariableType>::SendMessages() {
//     static std::vector<int> sendDataSizesBuffers;
//     static std::vector<std::vector<double>> sendMessageBuffers;

//     auto sendMessages = CreateMessages();

//     // メッセージを送信する
//     MPI_Request request;
//     for (auto&& [dest, message] : sendMessages) {
//         sendDataSizesBuffers.push_back(message.size());
//         MPI_Isend(&sendDataSizesBuffers.back(), 1, MPI_INT, dest,
//         dataSizeTag, MPI_COMM_WORLD, &request);
//         sendMessageBuffers.push_back(message);
//         MPI_Isend(sendMessageBuffers.back().data(),
//         sendDataSizesBuffers.back(), MPI_DOUBLE, dest, messageTag,
//         MPI_COMM_WORLD,
//                   &request);
//     }
// }

template <typename DecisionVariableType>
std::vector<std::vector<double>>
MpMoeadIdealTopology<DecisionVariableType>::ReceiveMessages() {
    std::vector<std::vector<double>> receiveMessages;
    for (auto&& source : neighboringRanks) {
        while (true) {
            MPI_Status status;
            int canReceive;
            MPI_Iprobe(source, messageTag, MPI_COMM_WORLD, &canReceive,
                       &status);
            if (!canReceive) {
                break;
            }

            int receiveDataSize;
            MPI_Get_count(&status, MPI_DOUBLE, &receiveDataSize);
            std::vector<double> receive(receiveDataSize);
            MPI_Recv(receive.data(), receiveDataSize, MPI_DOUBLE, source,
                     messageTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            receiveMessages.push_back(std::move(receive));
        }
    }

    for (auto&& source : idealTopologyToReceive) {
        while (true) {
            MPI_Status status;
            int canReceive;
            MPI_Iprobe(source, messageTag, MPI_COMM_WORLD, &canReceive,
                       &status);
            if (!canReceive) {
                break;
            }

            int receiveDataSize;
            MPI_Get_count(&status, MPI_DOUBLE, &receiveDataSize);
            std::vector<double> receive(receiveDataSize);
            MPI_Recv(receive.data(), receiveDataSize, MPI_DOUBLE, source,
                     messageTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            receiveMessages.push_back(std::move(receive));
        }
    }

    return receiveMessages;
}

template <typename DecisionVariableType>
void MpMoeadIdealTopology<DecisionVariableType>::UpdateWithMessage(
    std::vector<double>& message) {
    int limit = message.size();
    bool containsIdealPoint =
        message.size() % singleMessageSize == objectivesNum;
    if (containsIdealPoint) {
        limit -= objectivesNum;
    }
    for (int i = 0; i < limit; i += singleMessageSize) {
        int index = message[i];
        if (!(IsInternal(index) || IsExternal(index))) {
            continue;
        }

        Eigen::ArrayX<DecisionVariableType> newSolution =
            Eigen::Map<Eigen::ArrayXd>(message.data() + i + 1,
                                       decisionVariablesNum);
        Eigen::ArrayXd newObjectives = Eigen::Map<Eigen::ArrayXd>(
            message.data() + i + 1 + decisionVariablesNum, objectivesNum);
        Individual<DecisionVariableType> newIndividual(
            std::move(newSolution), std::move(newObjectives));
        if (IsExternal(index)) {
            clonedExternalIndividuals[index].UpdateFrom(newIndividual);
        } else {
            double newSubObjective = decomposition->ComputeObjective(
                individuals[index].weightVector, newIndividual.objectives);
            double oldSubObjective = decomposition->ComputeObjective(
                individuals[index].weightVector, individuals[index].objectives);
            if (newSubObjective < oldSubObjective) {
                individuals[index].UpdateFrom(newIndividual);
                updatedSolutionIndexes.insert(index);
            }
        }

        UpdateIdealPoint(newIndividual.objectives);
    }

    if (containsIdealPoint) {
        UpdateIdealPointWithMessage(message);
    }
}

}  // namespace Eacpp
