#pragma once

#include <mpi.h>

#include <algorithm>
#include <eigen3/Eigen/Core>
#include <fstream>
#include <memory>
#include <numeric>
#include <ranges>
#include <set>
#include <sstream>
#include <string>
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
constexpr int messageTag = 0;

// TODO: vectorにおいてできるだけreserveを使ってメモリ確保を行う
template <typename DecisionVariableType>
class NtMoead : public IMoead<DecisionVariableType> {
   public:
    NtMoead(int generationNum, int neighborhoodSize, int divisionsNumOfWeightVector, int migrationInterval,
            const std::shared_ptr<ICrossover<DecisionVariableType>>& crossover,
            const std::shared_ptr<IDecomposition>& decomposition,
            const std::shared_ptr<IMutation<DecisionVariableType>>& mutation,
            const std::shared_ptr<IProblem<DecisionVariableType>>& problem,
            const std::shared_ptr<IRepair<DecisionVariableType>>& repair,
            const std::shared_ptr<ISampling<DecisionVariableType>>& sampling, const std::shared_ptr<ISelection>& selection)
        : generationNum(generationNum),
          neighborhoodSize(neighborhoodSize),
          divisionsNumOfWeightVector(divisionsNumOfWeightVector),
          migrationInterval(migrationInterval) {
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
    ~NtMoead() {}

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
    std::set<int> updatedExternalSolutionIndexes;
    std::unordered_map<int, Individual<DecisionVariableType>> individuals;

    std::unordered_map<int, std::vector<int>> rankIndexesToSend;

    void Clear();
    std::vector<std::vector<int>> ReadAdjacencyList();
    void CalculateRankIndexesByNode(std::vector<int>& allRankIndexes, std::vector<int>& sizesAllRankIndexes,
                                    std::vector<int>& numsRankIndexes, std::vector<int>& sizesNumRank);
    void CalculateRankIndexesToSend(std::vector<int>& allRankIndexes, std::vector<int>& numsRankIndexes);
    std::pair<std::vector<int>, std::vector<int>> GenerateExternalNeighborhood(std::vector<int>& neighborhoodIndexes,
                                                                               std::vector<int>& populationSizes);
    std::vector<double> GetWeightVectorsMatchingIndexes(std::vector<double>& weightVectors, std::vector<int>& indexes);
    std::set<int> CalculateNeighboringRanks();
    void InitializeIndividualAndWeightVector(std::vector<Eigen::ArrayXd>& weightVectors,
                                             std::vector<std::vector<int>>& neighborhoodIndexes,
                                             std::vector<Eigen::ArrayXd>& externalNeighboringWeightVectors);
    void CalculateRanksToSent(const std::vector<int>& neighborhoodIndexes, const std::vector<int>& populationSizes,
                              std::vector<int>& outRanksToSentByRank, std::vector<int>& outSizes);
    std::vector<std::vector<double>> ScatterPopulation(const std::vector<int>& ranksToSend,
                                                       const std::set<int>& neighboringRanks);

    void InitializePopulation();
    void InitializeExternalPopulation(std::vector<std::vector<double>>& receivedIndividuals);
    void InitializeIdealPoint();
    std::vector<Individual<DecisionVariableType>> SelectParents(int index);
    Individual<DecisionVariableType> GenerateNewIndividual(int index);
    void UpdateNeighboringIndividuals(int index, Individual<DecisionVariableType>& newIndividual);
    bool IsInternal(int index);
    bool IsExternal(int index);
    bool HasIndividual(int index);

    std::unordered_map<int, std::vector<double>> CreateMessages();
    void SendMessages();
    std::vector<std::vector<double>> ReceiveMessages();
    void UpdateWithMessage(std::vector<double>& message);

#ifdef _TEST_
   public:
    NtMoead(int generationNum, int neighborhoodSize, int divisionsNumOfWeightVector, int migrationInterval,
            int decisionVariablesNum, int objectivesNum)
        : generationNum(generationNum),
          neighborhoodSize(neighborhoodSize),
          divisionsNumOfWeightVector(divisionsNumOfWeightVector),
          migrationInterval(migrationInterval),
          decisionVariablesNum(decisionVariablesNum),
          objectivesNum(objectivesNum),
          currentGeneration(0) {}

    friend class NtMoeadTest;
    friend class NtMoeadTestM;
#endif
};

template <typename DecisionVariableType>
void NtMoead<DecisionVariableType>::Run() {
    Initialize();

    while (!IsEnd()) {
        Update();
    }
}

template <typename DecisionVariableType>
void NtMoead<DecisionVariableType>::Initialize() {
    Clear();
    totalPopulationSize = initializer.CalculatePopulationSize(divisionsNumOfWeightVector, objectivesNum);
    InitializeMpi();
    InitializeIsland();
    InitializeIdealPoint();
    currentGeneration = 0;
}

template <typename DecisionVariableType>
void NtMoead<DecisionVariableType>::Update() {
    for (auto&& i : internalIndexes) {
        Individual<DecisionVariableType> newIndividual = GenerateNewIndividual(i);
        repair->Repair(newIndividual);
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
        updatedExternalSolutionIndexes.clear();
        for (auto&& message : messages) {
            UpdateWithMessage(message);
        }
    }
}

template <typename DecisionVariableType>
bool NtMoead<DecisionVariableType>::IsEnd() const {
    return currentGeneration >= generationNum;
}

template <typename DecisionVariableType>
std::vector<Eigen::ArrayXd> NtMoead<DecisionVariableType>::GetObjectivesList() const {
    std::vector<Eigen::ArrayXd> objectives;
    for (auto&& i : internalIndexes) {
        objectives.push_back(individuals.at(i).objectives);
    }

    return objectives;
}

template <typename DecisionVariableType>
std::vector<Eigen::ArrayX<DecisionVariableType>> NtMoead<DecisionVariableType>::GetSolutionList() const {
    std::vector<Eigen::ArrayX<DecisionVariableType>> solutions;
    for (auto&& i : internalIndexes) {
        solutions.push_back(individuals.at(i).solution);
    }

    return solutions;
}

template <typename DecisionVariableType>
void NtMoead<DecisionVariableType>::Clear() {
    internalIndexes.clear();
    externalIndexes.clear();
    updatedSolutionIndexes.clear();
    updatedExternalSolutionIndexes.clear();
    individuals.clear();
}

template <typename DecisionVariableType>
void NtMoead<DecisionVariableType>::InitializeMpi() {
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(nullptr, nullptr);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &parallelSize);
}

template <typename DecisionVariableType>
void NtMoead<DecisionVariableType>::InitializeIsland() {
    std::vector<double> weightVectors1d;
    std::vector<int> neighborhoodIndexes1d;
    std::vector<int> populationSizes;
    if (rank == 0) {
        initializer.GenerateWeightVectorsAndNeighborhoods(divisionsNumOfWeightVector, objectivesNum, neighborhoodSize,
                                                          weightVectors1d, neighborhoodIndexes1d);
        populationSizes = CalculateNodeWorkloads(totalPopulationSize, parallelSize);
    }

    // 重みベクトルと近傍インデックスを分散
    std::vector<double> receivedWeightVectors = Scatterv(weightVectors1d, populationSizes, objectivesNum, rank, parallelSize);
    std::vector<int> receivedNeighborhoodIndexes =
        Scatterv(neighborhoodIndexes1d, populationSizes, neighborhoodSize, rank, parallelSize);

    // ノード全体の近傍のインデックスと重みベクトルを生成
    std::vector<int> noduplicateNeighborhoodIndexes;
    std::vector<int> neighborhoodSizes;
    std::vector<double> sendExternalNeighboringWeightVectors;
    if (rank == 0) {
        std::tie(noduplicateNeighborhoodIndexes, neighborhoodSizes) =
            GenerateExternalNeighborhood(neighborhoodIndexes1d, populationSizes);
        sendExternalNeighboringWeightVectors = GetWeightVectorsMatchingIndexes(weightVectors1d, noduplicateNeighborhoodIndexes);
    }

    // ノード全体の近傍のインデックスと重みベクトルを分散
    externalIndexes = Scatterv(noduplicateNeighborhoodIndexes, neighborhoodSizes, 1, rank, parallelSize);
    std::vector<double> receivedExternalNeighboringWeightVectors =
        Scatterv(sendExternalNeighboringWeightVectors, neighborhoodSizes, objectivesNum, rank, parallelSize);

    // 隣接ノードに送信する
    std::vector<int> allRankIndexes;
    std::vector<int> sizesAllRankIndexes;
    std::vector<int> numsRankIndexes;
    std::vector<int> sizesNumRank;
    if (rank == 0) {
        CalculateRankIndexesByNode(allRankIndexes, sizesAllRankIndexes, numsRankIndexes, sizesNumRank);
    }
    std::vector<int> receivedRankIndexes = Scatterv(allRankIndexes, sizesAllRankIndexes, 1, rank, parallelSize);
    std::vector<int> receivedNumsRankIndexes = Scatterv(numsRankIndexes, sizesNumRank, 1, rank, parallelSize);
    CalculateRankIndexesToSend(allRankIndexes, numsRankIndexes);

    // 受信したデータを2Dに変換
    std::vector<Eigen::ArrayXd> weightVectors = TransformToEigenArrayX2d(receivedWeightVectors, objectivesNum);
    std::vector<std::vector<int>> neighborhoodIndexes = TransformTo2d(receivedNeighborhoodIndexes, neighborhoodSize);
    std::vector<Eigen::ArrayXd> externalNeighboringWeightVectors =
        TransformToEigenArrayX2d(receivedExternalNeighboringWeightVectors, objectivesNum);

    internalIndexes = GenerateNodeIndexes(totalPopulationSize, rank, parallelSize);
    InitializePopulation();

    // 送信するランク，受信するランクを計算
    std::vector<int> sendRanksToSentByRank;
    std::vector<int> ranksToSentByRankSizes;
    if (rank == 0) {
        CalculateRanksToSent(neighborhoodIndexes1d, populationSizes, sendRanksToSentByRank, ranksToSentByRankSizes);
    }
    std::vector<int> ranksToSend = Scatterv(sendRanksToSentByRank, ranksToSentByRankSizes, 1, rank, parallelSize);
    std::set<int> neighboringRanks = CalculateNeighboringRanks();

    auto receivedExternalIndividuals = ScatterPopulation(ranksToSend, neighboringRanks);

    InitializeExternalPopulation(receivedExternalIndividuals);
    InitializeIndividualAndWeightVector(weightVectors, neighborhoodIndexes, externalNeighboringWeightVectors);
}

template <typename DecisionVariableType>
std::vector<std::vector<int>> NtMoead<DecisionVariableType>::ReadAdjacencyList() {
    constexpr const char* filename = "data/graph/hoffmanSingletonGraphAdjacencyList.csv";

    std::ifstream ifs(filename);
    if (!ifs) {
        std::cerr << "Failed to open " << filename << std::endl;
        std::exit(1);
    }

    std::vector<std::vector<int>> adjacencyList;
    adjacencyList.reserve(parallelSize);
    std::string line;
    while (std::getline(ifs, line)) {
        std::vector<int> neighbors;
        std::istringstream iss(line);
        std::string neighborStr;
        while (std::getline(iss, neighborStr, ',')) {
            int neighbor = std::stoi(neighborStr);
            neighbors.push_back(neighbor);
        }

        adjacencyList.push_back(neighbors);
    }

    return adjacencyList;
}

/// @brief 各ノードにおいて，通信対象のノードと送信するインデックスを計算する
/// @tparam DecisionVariableType
/// @param allRankIndexes {{rank1, index1, index2, rank2, index3, ...}, {rank3, index4, ...}, ...}
/// @param sizesAllRankIndexes {allSizeOfNode1, allSizeOfNode2, ...}
/// @param numsRankIndexes {{numOfRank1, numOfRank2, ...}, {numOfRank3, ...}, ...} rank number are not included
/// @param sizesNumRank {sizeOfNumOfNode1, sizeOfNumOfNode2, ...} equal to degrees of nodes
template <typename DecisionVariableType>
void NtMoead<DecisionVariableType>::CalculateRankIndexesByNode(std::vector<int>& allRankIndexes,
                                                               std::vector<int>& sizesAllRankIndexes,
                                                               std::vector<int>& numsRankIndexes,
                                                               std::vector<int>& sizesNumRank) {
    auto adjacencyList = ReadAdjacencyList();
    // TODO: あるノードが必要としている解は，自分の解とその近傍の解であるため，近傍も含めて計算する
    auto allNodeIndexes = GenerateAllNodeIndexes(totalPopulationSize, parallelSize);

    for (int rank = 0; rank < adjacencyList.size(); rank++) {
        sizesNumRank.push_back(adjacencyList[rank].size());

        int size = 0;
        for (auto&& neighbor : adjacencyList[rank]) {
            std::set<int> individualIndexes;
            individualIndexes.insert(allNodeIndexes[neighbor].begin(), allNodeIndexes[neighbor].end());
            for (auto&& k : adjacencyList[neighbor]) {
                if (k == rank) {
                    continue;
                }

                individualIndexes.insert(allNodeIndexes[k].begin(), allNodeIndexes[k].end());
            }
            allRankIndexes.push_back(neighbor);
            allRankIndexes.insert(allRankIndexes.end(), individualIndexes.begin(), individualIndexes.end());
            numsRankIndexes.push_back(individualIndexes.size());
            size += individualIndexes.size();
        }

        sizesAllRankIndexes.push_back(size + adjacencyList[rank].size());
    }
}

template <typename DecisionVariableType>
void NtMoead<DecisionVariableType>::CalculateRankIndexesToSend(std::vector<int>& allRankIndexes,
                                                               std::vector<int>& numsRankIndexes) {
    for (int i = 0, count = 0; i < allRankIndexes.size(); i += numsRankIndexes[count] + 1, ++count) {
        int rank = allRankIndexes[i];
        rankIndexesToSend[rank].insert(rankIndexesToSend[rank].end(), allRankIndexes.begin() + i + 1,
                                       allRankIndexes.begin() + i + 1 + numsRankIndexes[count]);
    }
}

template <typename DecisionVariableType>
std::pair<std::vector<int>, std::vector<int>> NtMoead<DecisionVariableType>::GenerateExternalNeighborhood(
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
std::vector<double> NtMoead<DecisionVariableType>::GetWeightVectorsMatchingIndexes(std::vector<double>& weightVectors,
                                                                                   std::vector<int>& indexes) {
    std::vector<double> matchingWeightVectors;
    for (int i = 0; i < indexes.size(); i++) {
        matchingWeightVectors.insert(matchingWeightVectors.end(), weightVectors.begin() + indexes[i] * objectivesNum,
                                     weightVectors.begin() + (indexes[i] + 1) * objectivesNum);
    }
    return matchingWeightVectors;
}

template <typename DecisionVariableType>
std::set<int> NtMoead<DecisionVariableType>::CalculateNeighboringRanks() {
    std::set<int> neighboringRanks;
    for (auto&& i : externalIndexes) {
        int rank = GetRankFromIndex(totalPopulationSize, i, parallelSize);
        neighboringRanks.insert(rank);
    }

    return neighboringRanks;
}

template <typename DecisionVariableType>
void NtMoead<DecisionVariableType>::InitializePopulation() {
    int sampleNum = internalIndexes.size();
    std::vector<Individual<DecisionVariableType>> sampledIndividuals = sampling->Sample(sampleNum, decisionVariablesNum);
    for (int i = 0; i < internalIndexes.size(); i++) {
        individuals[internalIndexes[i]] = sampledIndividuals[i];
        problem->ComputeObjectiveSet(individuals[internalIndexes[i]]);
    }
}

template <typename DecisionVariableType>
void NtMoead<DecisionVariableType>::InitializeExternalPopulation(std::vector<std::vector<double>>& receivedIndividuals) {
    for (auto&& receive : receivedIndividuals) {
        for (int i = 0; i < receive.size(); i += decisionVariablesNum + objectivesNum + 1) {
            int index = receive[i];
            if (IsExternal(index)) {
                individuals[index].solution = Eigen::Map<Eigen::ArrayXd>(receive.data() + (i + 1), decisionVariablesNum);
                individuals[index].objectives =
                    Eigen::Map<Eigen::ArrayXd>(receive.data() + (i + 1 + decisionVariablesNum), objectivesNum);
            }
        }
    }
}

template <typename DecisionVariableType>
void NtMoead<DecisionVariableType>::CalculateRanksToSent(const std::vector<int>& neighborhoodIndexes,
                                                         const std::vector<int>& populationSizes,
                                                         std::vector<int>& outRanksToSentByRank, std::vector<int>& outSizes) {
    std::vector<std::set<int>> ranksToSentByRank(parallelSize, std::set<int>());
    for (int dest = 0, count = 0; dest < parallelSize; count += populationSizes[dest] * neighborhoodSize, ++dest) {
        std::vector<int> neighborhood;
        std::copy(neighborhoodIndexes.begin() + count,
                  neighborhoodIndexes.begin() + count + populationSizes[dest] * neighborhoodSize,
                  std::back_inserter(neighborhood));
        std::sort(neighborhood.begin(), neighborhood.end());
        neighborhood.erase(std::unique(neighborhood.begin(), neighborhood.end()), neighborhood.end());

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
std::vector<std::vector<double>> NtMoead<DecisionVariableType>::ScatterPopulation(const std::vector<int>& ranksToSend,
                                                                                  const std::set<int>& neighboringRanks) {
    std::vector<double> individualsToSend;
    individualsToSend.reserve(internalIndexes.size() * (decisionVariablesNum + objectivesNum + 1));
    for (auto&& i : internalIndexes) {
        individualsToSend.push_back(i);
        individualsToSend.insert(individualsToSend.end(), individuals[i].solution.begin(), individuals[i].solution.end());
        individualsToSend.insert(individualsToSend.end(), individuals[i].objectives.begin(), individuals[i].objectives.end());
    }
    int dataSize = individualsToSend.size();

    std::vector<MPI_Request> requests;
    requests.reserve(ranksToSend.size());
    for (auto&& i : ranksToSend) {
        requests.emplace_back();
        MPI_Isend(individualsToSend.data(), individualsToSend.size(), MPI_DOUBLE, i, messageTag, MPI_COMM_WORLD,
                  &requests.back());
    }

    std::vector<std::vector<double>> receivedIndividuals;
    receivedIndividuals.reserve(neighboringRanks.size());
    for (auto&& rank : neighboringRanks) {
        MPI_Status status;
        MPI_Probe(rank, messageTag, MPI_COMM_WORLD, &status);
        int receivedDataSize;
        MPI_Get_count(&status, MPI_DOUBLE, &receivedDataSize);
        receivedIndividuals.emplace_back(receivedDataSize);
        MPI_Recv(receivedIndividuals.back().data(), receivedDataSize, MPI_DOUBLE, rank, messageTag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }

    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

    return receivedIndividuals;
}

template <typename DecisionVariableType>
void NtMoead<DecisionVariableType>::InitializeIndividualAndWeightVector(
    std::vector<Eigen::ArrayXd>& weightVectors, std::vector<std::vector<int>>& neighborhoodIndexes,
    std::vector<Eigen::ArrayXd>& externalNeighboringWeightVectors) {
    for (int i = 0; i < internalIndexes.size(); i++) {
        individuals[internalIndexes[i]].weightVector = std::move(weightVectors[i]);
        individuals[internalIndexes[i]].neighborhood = std::move(neighborhoodIndexes[i]);
    }
    for (int i = 0; i < externalIndexes.size(); i++) {
        individuals[externalIndexes[i]].weightVector = std::move(externalNeighboringWeightVectors[i]);
    }
}

template <typename DecisionVariableType>
void NtMoead<DecisionVariableType>::InitializeIdealPoint() {
    for (auto&& [_, individual] : individuals) {
        decomposition->UpdateIdealPoint(individual.objectives);
    }
}

template <typename DecisionVariableType>
std::vector<Individual<DecisionVariableType>> NtMoead<DecisionVariableType>::SelectParents(int index) {
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
Individual<DecisionVariableType> NtMoead<DecisionVariableType>::GenerateNewIndividual(int index) {
    std::vector<Individual<DecisionVariableType>> parents = SelectParents(index);
    Individual<DecisionVariableType> newIndividual = crossover->Cross(parents);
    mutation->Mutate(newIndividual);
    return newIndividual;
}

template <typename DecisionVariableType>
void NtMoead<DecisionVariableType>::UpdateNeighboringIndividuals(int index, Individual<DecisionVariableType>& newIndividual) {
    for (auto&& i : individuals[index].neighborhood) {
        double newSubObjective = decomposition->ComputeObjective(individuals[i].weightVector, newIndividual.objectives);
        double oldSubObjective = decomposition->ComputeObjective(individuals[i].weightVector, individuals[i].objectives);
        if (newSubObjective < oldSubObjective) {
            individuals[i].UpdateFrom(newIndividual);
            if (IsInternal(i)) {
                updatedSolutionIndexes.insert(i);
            } else {
                updatedExternalSolutionIndexes.insert(i);
            }
        }
    }
}

template <typename DecisionVariableType>
bool NtMoead<DecisionVariableType>::IsInternal(int index) {
    return std::find(internalIndexes.begin(), internalIndexes.end(), index) != internalIndexes.end();
}

template <typename DecisionVariableType>
bool NtMoead<DecisionVariableType>::IsExternal(int index) {
    return std::find(externalIndexes.begin(), externalIndexes.end(), index) != externalIndexes.end();
}

template <typename DecisionVariableType>
bool NtMoead<DecisionVariableType>::HasIndividual(int index) {
    return individuals.find(index) != individuals.end();
}

template <typename DecisionVariableType>
std::unordered_map<int, std::vector<double>> NtMoead<DecisionVariableType>::CreateMessages() {
    std::vector<double> updatedInternalIndividuals;
    for (auto&& i : updatedSolutionIndexes) {
        updatedInternalIndividuals.push_back(i);
        updatedInternalIndividuals.insert(updatedInternalIndividuals.end(), individuals[i].solution.begin(),
                                          individuals[i].solution.end());
        updatedInternalIndividuals.insert(updatedInternalIndividuals.end(), individuals[i].objectives.begin(),
                                          individuals[i].objectives.end());
    }

    std::unordered_map<int, std::vector<double>> dataToSend;
    for (auto&& [rank, indexes] : rankIndexesToSend) {
        dataToSend[rank] = std::vector<double>();
        for (auto&& i : indexes) {
            if (!HasIndividual(i)) {
                continue;
            }

            dataToSend[rank].push_back(i);
            dataToSend[rank].insert(dataToSend[rank].end(), individuals[i].solution.begin(), individuals[i].solution.end());
            dataToSend[rank].insert(dataToSend[rank].end(), individuals[i].objectives.begin(), individuals[i].objectives.end());
        }
    }

    // for (auto&& [rank, message] : dataToSend) {
    //     message.insert(message.end(), updatedInternalIndividuals.begin(), updatedInternalIndividuals.end());
    // }

    // for (auto&& i : ranksToSend) {
    //     dataToSend[i].insert(dataToSend[i].end(), updatedInternalIndividuals.begin(), updatedInternalIndividuals.end());
    // }

    // int special = neighborhoodSize / CalculateNodeWorkload(totalPopulationSize, rank, parallelSize);
    // if (rank == special) {
    //     constexpr int dest = 0;
    //     dataToSend[dest] = updatedInternalIndividuals;
    // } else if (rank == parallelSize - special - 1) {
    //     int dest = parallelSize - 1;
    //     dataToSend[dest] = updatedInternalIndividuals;
    // }

    return dataToSend;
}

template <typename DecisionVariableType>
void NtMoead<DecisionVariableType>::SendMessages() {
    // // MPI_Isendで使うバッファ
    // static std::array<std::unordered_map<int, std::vector<double>>, maxBufferSize> sendMessageBuffers;
    // static int sendMessageBufferIndex = 0;
    // sendMessageBufferIndex = (sendMessageBufferIndex + 1) % maxBufferSize;

    // std::unordered_map<int, std::vector<double>>& sendMessages = sendMessageBuffers[sendMessageBufferIndex];
    // sendMessages = CreateMessages();

    // // メッセージを送信する
    // MPI_Request request;
    // for (auto&& [dest, message] : sendMessages) {
    //     MPI_Isend(message.data(), message.size(), MPI_DOUBLE, dest, messageTag, MPI_COMM_WORLD, &request);
    // }
}

// bufferが無限バージョン
// template <typename DecisionVariableType>
// void NtMoead<DecisionVariableType>::SendMessages() {
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
std::vector<std::vector<double>> NtMoead<DecisionVariableType>::ReceiveMessages() {
    std::vector<std::vector<double>> receiveMessages;
    // for (auto&& source : neighboringRanks) {
    //     while (true) {
    //         MPI_Status status;
    //         int canReceive;
    //         MPI_Iprobe(source, messageTag, MPI_COMM_WORLD, &canReceive, &status);
    //         if (!canReceive) {
    //             break;
    //         }

    //         int receiveDataSize;
    //         MPI_Get_count(&status, MPI_DOUBLE, &receiveDataSize);
    //         std::vector<double> receive = std::vector<double>(receiveDataSize);
    //         MPI_Recv(receive.data(), receiveDataSize, MPI_DOUBLE, source, messageTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //         receiveMessages.push_back(std::move(receive));
    //     }
    // }

    return receiveMessages;
}

template <typename DecisionVariableType>
void NtMoead<DecisionVariableType>::UpdateWithMessage(std::vector<double>& message) {
    // for (int i = 0; i < message.size(); i += decisionVariablesNum + 1) {
    //     int index = message[i];
    //     if (!(IsInternal(index) || IsExternal(index))) {
    //         continue;
    //     }

    //     Eigen::ArrayX<DecisionVariableType> newSolution =
    //         Eigen::Map<Eigen::ArrayX<DecisionVariableType>>(message.data() + i + 1, decisionVariablesNum);
    //     Individual<DecisionVariableType> newIndividual(newSolution);
    //     problem->ComputeObjectiveSet(newIndividual);
    //     if (IsExternal(index)) {
    //         clonedExternalIndividuals[index].UpdateFrom(newIndividual);
    //     } else {
    //         double newSubObjective = decomposition->ComputeObjective(individuals[index].weightVector,
    //         newIndividual.objectives); double oldSubObjective =
    //             decomposition->ComputeObjective(individuals[index].weightVector, individuals[index].objectives);
    //         if (newSubObjective < oldSubObjective) {
    //             individuals[index].UpdateFrom(newIndividual);
    //             updatedSolutionIndexes.insert(index);
    //         }
    //     }

    //     decomposition->UpdateIdealPoint(newIndividual.objectives);
    // }
}

}  // namespace Eacpp
