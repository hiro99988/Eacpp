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
#include "Utils/MpiUtils.h"
#include "Utils/Utils.h"

namespace Eacpp {

template <typename DecisionVariableType>
class OneNtMoead : public IMoead<DecisionVariableType> {
   public:
    constexpr static int maxBufferSize = 100;
    constexpr static int messageTag = 0;

   private:
    int totalPopulationSize;
    int generationNum;
    int currentGeneration;
    int decisionVariablesNum;
    int objectivesNum;
    int neighborhoodSize;
    int migrationInterval;
    int divisionsNumOfWeightVector;
    std::string adjacencyListFilePath;
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
    std::set<int> updatedExternalIndexes;
    std::unordered_map<int, Individual<DecisionVariableType>> individuals;
    std::unordered_map<int, std::vector<int>> rankAndExternalIndexesToSend;
    std::set<int> neighboringRanks;
    int singleMessageSize;
    bool idealPointMigration;
    bool isIdealPointUpdated;

   public:
    OneNtMoead(int generationNum, int neighborhoodSize, int divisionsNumOfWeightVector, int migrationInterval,
               std::string adjacencyListFilePath, const std::shared_ptr<ICrossover<DecisionVariableType>>& crossover,
               const std::shared_ptr<IDecomposition>& decomposition,
               const std::shared_ptr<IMutation<DecisionVariableType>>& mutation,
               const std::shared_ptr<IProblem<DecisionVariableType>>& problem,
               const std::shared_ptr<IRepair<DecisionVariableType>>& repair,
               const std::shared_ptr<ISampling<DecisionVariableType>>& sampling, const std::shared_ptr<ISelection>& selection,
               bool idealPointMigration = false)
        : generationNum(generationNum),
          neighborhoodSize(neighborhoodSize),
          migrationInterval(migrationInterval),
          divisionsNumOfWeightVector(divisionsNumOfWeightVector),
          adjacencyListFilePath(adjacencyListFilePath) {
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
        this->idealPointMigration = idealPointMigration;
        decisionVariablesNum = problem->DecisionVariablesNum();
        objectivesNum = problem->ObjectivesNum();
        currentGeneration = 0;
        singleMessageSize = decisionVariablesNum + objectivesNum + 1;
        isIdealPointUpdated = false;
    }
    virtual ~OneNtMoead() {}

    int CurrentGeneration() const override {
        return currentGeneration;
    }

    void Run() override {
        Initialize();

        while (!IsEnd()) {
            Update();
        }
    }

    void Initialize() override {
        Clear();
        InitializeMpi();
        totalPopulationSize = initializer.CalculatePopulationSize(divisionsNumOfWeightVector, objectivesNum);
        decomposition->InitializeIdealPoint(objectivesNum);
        InitializeTopology();
        InitializeIsland();
        currentGeneration = 0;
    }

    void Update() override {
        for (auto&& i : internalIndexes) {
            Individual<DecisionVariableType> newIndividual = GenerateNewIndividual(i);
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
            updatedExternalIndexes.clear();
            isIdealPointUpdated = false;

            for (auto&& message : messages) {
                UpdateWithMessage(message);
            }
        }
    }

    bool IsEnd() const override {
        return currentGeneration >= generationNum;
    }

    std::vector<Eigen::ArrayXd> GetObjectivesList() const override {
        std::vector<Eigen::ArrayXd> objectives;
        for (auto&& i : internalIndexes) {
            objectives.push_back(individuals.at(i).objectives);
        }

        return objectives;
    }

    std::vector<Eigen::ArrayX<DecisionVariableType>> GetSolutionList() const override {
        std::vector<Eigen::ArrayX<DecisionVariableType>> solutions;
        for (auto&& i : internalIndexes) {
            solutions.push_back(individuals.at(i).solution);
        }

        return solutions;
    }

    void InitializeMpi() {
        int initialized;
        MPI_Initialized(&initialized);
        if (!initialized) {
            MPI_Init(nullptr, nullptr);
        }
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &parallelSize);
    }

    void InitializeIsland() {
        std::vector<double> weightVectors1d;
        std::vector<int> neighborhoodIndexes1d;
        std::vector<int> populationSizes;
        std::vector<int> noduplicateNeighborhoodIndexes;
        std::vector<int> neighborhoodSizes;
        std::vector<double> sendExternalNeighboringWeightVectors;
        std::vector<int> sendRanksToSentByRank;
        std::vector<int> ranksToSentByRankSizes;
        if (rank == 0) {
            initializer.GenerateWeightVectorsAndNeighborhoods(divisionsNumOfWeightVector, objectivesNum, neighborhoodSize,
                                                              weightVectors1d, neighborhoodIndexes1d);
            populationSizes = CalculateNodeWorkloads(totalPopulationSize, parallelSize);
        }

        std::vector<double> receivedWeightVectors =
            Scatterv(weightVectors1d, populationSizes, objectivesNum, rank, parallelSize);
        std::vector<int> receivedNeighborhoodIndexes =
            Scatterv(neighborhoodIndexes1d, populationSizes, neighborhoodSize, rank, parallelSize);

        if (rank == 0) {
            std::tie(noduplicateNeighborhoodIndexes, neighborhoodSizes) =
                GenerateExternalNeighborhood(neighborhoodIndexes1d, populationSizes);
            sendExternalNeighboringWeightVectors =
                GetWeightVectorsMatchingIndexes(weightVectors1d, noduplicateNeighborhoodIndexes);
        }

        externalIndexes = Scatterv(noduplicateNeighborhoodIndexes, neighborhoodSizes, 1, rank, parallelSize);
        std::vector<double> receivedExternalNeighboringWeightVectors =
            Scatterv(sendExternalNeighboringWeightVectors, neighborhoodSizes, objectivesNum, rank, parallelSize);

        if (rank == 0) {
            CalculateRanksToSent(neighborhoodIndexes1d, populationSizes, sendRanksToSentByRank, ranksToSentByRankSizes);
        }
        auto ranksToSend = Scatterv(sendRanksToSentByRank, ranksToSentByRankSizes, 1, rank, parallelSize);

        std::vector<Eigen::ArrayXd> weightVectors = TransformToEigenArrayX2d(receivedWeightVectors, objectivesNum);
        std::vector<std::vector<int>> neighborhoodIndexes = TransformTo2d(receivedNeighborhoodIndexes, neighborhoodSize);
        std::vector<Eigen::ArrayXd> externalNeighboringWeightVectors =
            TransformToEigenArrayX2d(receivedExternalNeighboringWeightVectors, objectivesNum);

        internalIndexes = GenerateNodeIndexes(totalPopulationSize, rank, parallelSize);
        auto ranksToReceive = CalculateNeighboringRanks();
        InitializePopulation();

        auto receivedExternalSolutions = ScatterPopulation(ranksToSend, ranksToReceive);

        InitializeExternalPopulation(receivedExternalSolutions);
        InitializeIndividualAndWeightVector(weightVectors, neighborhoodIndexes, externalNeighboringWeightVectors);
    }

   private:
    void Clear() {
        internalIndexes.clear();
        externalIndexes.clear();
        updatedSolutionIndexes.clear();
        updatedExternalIndexes.clear();
        individuals.clear();
        rankAndExternalIndexesToSend.clear();
        neighboringRanks.clear();
    }

    void InitializeTopology() {
        std::ifstream ifs = OpenInputFile(adjacencyListFilePath);
        std::string line;
        for (int i = 0; i <= rank; ++i) {
            if (!std::getline(ifs, line)) {
                throw std::runtime_error("Not enough lines in adjacencyListFile");
            }
        }
        std::stringstream ss(line);
        std::string item;
        while (std::getline(ss, item, ',')) {
            int rank = std::stoi(item);
            neighboringRanks.insert(rank);
        }
    }

    std::pair<std::vector<int>, std::vector<int>> GenerateExternalNeighborhood(std::vector<int>& neighborhoodIndexes,
                                                                               std::vector<int>& populationSizes) {
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

    std::vector<double> GetWeightVectorsMatchingIndexes(std::vector<double>& weightVectors, std::vector<int>& indexes) {
        std::vector<double> matchingWeightVectors;
        for (int i = 0; i < indexes.size(); i++) {
            matchingWeightVectors.insert(matchingWeightVectors.end(), weightVectors.begin() + indexes[i] * objectivesNum,
                                         weightVectors.begin() + (indexes[i] + 1) * objectivesNum);
        }
        return matchingWeightVectors;
    }

    void CalculateRanksToSent(const std::vector<int>& neighborhoodIndexes, const std::vector<int>& populationSizes,
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

    std::set<int> CalculateNeighboringRanks() {
        std::set<int> ranksToReceive;
        for (auto&& i : externalIndexes) {
            int rank = GetRankFromIndex(totalPopulationSize, i, parallelSize);
            ranksToReceive.insert(rank);
            if (neighboringRanks.find(rank) == neighboringRanks.end()) {
                continue;
            }

            rankAndExternalIndexesToSend[rank].push_back(i);
        }

        return ranksToReceive;
    }

    std::vector<std::vector<double>> ScatterPopulation(const std::vector<int>& ranksToSend,
                                                       const std::set<int>& ranksToReceive) {
        std::vector<double> individualsToSend;
        individualsToSend.reserve(idealPointMigration ? internalIndexes.size() * singleMessageSize + objectivesNum
                                                      : internalIndexes.size() * singleMessageSize);
        for (auto&& i : internalIndexes) {
            individualsToSend.push_back(i);
            individualsToSend.insert(individualsToSend.end(), individuals[i].solution.begin(), individuals[i].solution.end());
            individualsToSend.insert(individualsToSend.end(), individuals[i].objectives.begin(),
                                     individuals[i].objectives.end());
        }

        if (idealPointMigration) {
            individualsToSend.insert(individualsToSend.end(), decomposition->IdealPoint().begin(),
                                     decomposition->IdealPoint().end());
        }

        std::vector<MPI_Request> requests;
        requests.reserve(ranksToSend.size());
        for (auto&& i : ranksToSend) {
            requests.emplace_back();
            MPI_Isend(individualsToSend.data(), individualsToSend.size(), MPI_DOUBLE, i, messageTag, MPI_COMM_WORLD,
                      &requests.back());
        }

        std::vector<std::vector<double>> receivedIndividuals;
        receivedIndividuals.reserve(ranksToReceive.size());
        for (auto&& rank : ranksToReceive) {
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

    void InitializeExternalPopulation(std::vector<std::vector<double>>& receivedSolutions) {
        for (auto&& receive : receivedSolutions) {
            int limit = idealPointMigration ? receive.size() - objectivesNum : receive.size();
            for (int i = 0; i < limit; i += singleMessageSize) {
                int index = receive[i];
                if (!IsExternal(index)) {
                    continue;
                }

                individuals[index].solution = Eigen::Map<Eigen::ArrayXd>(receive.data() + (i + 1), decisionVariablesNum);
                individuals[index].objectives =
                    Eigen::Map<Eigen::ArrayXd>(receive.data() + (i + 1 + decisionVariablesNum), objectivesNum);
                UpdateIdealPoint(individuals[index].objectives);
            }

            if (idealPointMigration) {
                UpdateIdealPointWithMessage(receive);
            }
        }
    }

    void InitializeIndividualAndWeightVector(std::vector<Eigen::ArrayXd>& weightVectors,
                                             std::vector<std::vector<int>>& neighborhoodIndexes,
                                             std::vector<Eigen::ArrayXd>& externalNeighboringWeightVectors) {
        for (int i = 0; i < internalIndexes.size(); i++) {
            individuals[internalIndexes[i]].weightVector = std::move(weightVectors[i]);
            individuals[internalIndexes[i]].neighborhood = std::move(neighborhoodIndexes[i]);
        }
        for (int i = 0; i < externalIndexes.size(); i++) {
            individuals[externalIndexes[i]].weightVector = std::move(externalNeighboringWeightVectors[i]);
        }
    }

    void InitializePopulation() {
        int sampleNum = internalIndexes.size();
        std::vector<Individual<DecisionVariableType>> sampledIndividuals = sampling->Sample(sampleNum, decisionVariablesNum);
        for (int i = 0; i < internalIndexes.size(); i++) {
            individuals[internalIndexes[i]] = sampledIndividuals[i];
            problem->ComputeObjectiveSet(individuals[internalIndexes[i]]);
            decomposition->UpdateIdealPoint(individuals[internalIndexes[i]].objectives);
        }
    }

    std::vector<Individual<DecisionVariableType>> SelectParents(int index) {
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

    Individual<DecisionVariableType> GenerateNewIndividual(int index) {
        std::vector<Individual<DecisionVariableType>> parents = SelectParents(index);
        Individual<DecisionVariableType> newIndividual = crossover->Cross(parents);
        mutation->Mutate(newIndividual);
        return newIndividual;
    }

    void UpdateNeighboringIndividuals(int index, Individual<DecisionVariableType>& newIndividual) {
        for (auto&& i : individuals[index].neighborhood) {
            double newSubObjective = decomposition->ComputeObjective(individuals[i].weightVector, newIndividual.objectives);
            double oldSubObjective = decomposition->ComputeObjective(individuals[i].weightVector, individuals[i].objectives);
            if (newSubObjective < oldSubObjective) {
                individuals[i].UpdateFrom(newIndividual);
                if (IsInternal(i)) {
                    updatedSolutionIndexes.insert(i);
                } else {
                    updatedExternalIndexes.insert(i);
                }
            }
        }
    }

    bool IsInternal(int index) {
        return std::find(internalIndexes.begin(), internalIndexes.end(), index) != internalIndexes.end();
    }

    bool IsExternal(int index) {
        return std::find(externalIndexes.begin(), externalIndexes.end(), index) != externalIndexes.end();
    }

    bool IsUpdated(int index) {
        return updatedSolutionIndexes.find(index) != updatedSolutionIndexes.end();
    }

    bool IsUpdatedExternal(int index) {
        return updatedExternalIndexes.find(index) != updatedExternalIndexes.end();
    }

    void UpdateIdealPoint(const Eigen::ArrayXd& objectives) {
        auto idealPoint = decomposition->IdealPoint();
        for (int i = 0; i < objectivesNum; i++) {
            if (objectives(i) < idealPoint(i)) {
                isIdealPointUpdated = true;
                decomposition->UpdateIdealPoint(objectives);
                break;
            }
        }
    }

    void UpdateIdealPointWithMessage(const std::vector<double>& message) {
        if (message.size() % singleMessageSize == objectivesNum) {
            Eigen::ArrayXd receivedIdealPoint =
                Eigen::Map<const Eigen::ArrayXd>(message.data() + (message.size() - objectivesNum), objectivesNum);
            UpdateIdealPoint(receivedIdealPoint);
        }
    }

    std::unordered_map<int, std::vector<double>> CreateMessages() {
        std::vector<double> updatedInternalIndividuals;
        updatedInternalIndividuals.reserve(updatedSolutionIndexes.size() * singleMessageSize);
        for (auto&& i : updatedSolutionIndexes) {
            updatedInternalIndividuals.push_back(i);
            updatedInternalIndividuals.insert(updatedInternalIndividuals.end(), individuals[i].solution.begin(),
                                              individuals[i].solution.end());
            updatedInternalIndividuals.insert(updatedInternalIndividuals.end(), individuals[i].objectives.begin(),
                                              individuals[i].objectives.end());
        }

        std::unordered_map<int, std::vector<double>> dataToSend;
        for (auto&& [rank, indexes] : rankAndExternalIndexesToSend) {
            for (auto&& index : indexes) {
                if (!IsUpdatedExternal(index)) {
                    continue;
                }

                dataToSend[rank].push_back(index);
                dataToSend[rank].insert(dataToSend[rank].end(), individuals[index].solution.begin(),
                                        individuals[index].solution.end());
                dataToSend[rank].insert(dataToSend[rank].end(), individuals[index].objectives.begin(),
                                        individuals[index].objectives.end());
            }
        }

        for (auto&& rank : neighboringRanks) {
            dataToSend[rank].insert(dataToSend[rank].end(), updatedInternalIndividuals.begin(),
                                    updatedInternalIndividuals.end());
            if (idealPointMigration && isIdealPointUpdated) {
                dataToSend[rank].insert(dataToSend[rank].end(), decomposition->IdealPoint().begin(),
                                        decomposition->IdealPoint().end());
            }
        }

        return dataToSend;
    }

    void SendMessages() {
        // MPI_Isendで使うバッファ
        static std::array<std::unordered_map<int, std::vector<double>>, maxBufferSize> sendMessageBuffers;
        static int sendMessageBufferIndex = 0;
        sendMessageBufferIndex = (sendMessageBufferIndex + 1) % maxBufferSize;

        std::unordered_map<int, std::vector<double>>& sendMessages = sendMessageBuffers[sendMessageBufferIndex];
        sendMessages = CreateMessages();

        // メッセージを送信する
        MPI_Request request;
        for (auto&& [dest, message] : sendMessages) {
            MPI_Isend(message.data(), message.size(), MPI_DOUBLE, dest, messageTag, MPI_COMM_WORLD, &request);
        }
    }

    std::vector<std::vector<double>> ReceiveMessages() {
        std::vector<std::vector<double>> receiveMessages;
        for (auto&& source : neighboringRanks) {
            while (true) {
                MPI_Status status;
                int canReceive;
                MPI_Iprobe(source, messageTag, MPI_COMM_WORLD, &canReceive, &status);
                if (!canReceive) {
                    break;
                }

                int receiveDataSize;
                MPI_Get_count(&status, MPI_DOUBLE, &receiveDataSize);
                std::vector<double> receive(receiveDataSize);
                MPI_Recv(receive.data(), receiveDataSize, MPI_DOUBLE, source, messageTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                receiveMessages.push_back(std::move(receive));
            }
        }

        return receiveMessages;
    }

    void UpdateWithMessage(std::vector<double>& message) {
        int limit = message.size();
        bool containsIdealPoint = idealPointMigration && message.size() % singleMessageSize == objectivesNum;
        if (containsIdealPoint) {
            limit -= objectivesNum;
        }

        for (int i = 0; i < limit; i += singleMessageSize) {
            int index = message[i];
            bool isInternal = IsInternal(index);
            if (!(isInternal || IsExternal(index))) {
                continue;
            }

            Eigen::ArrayX<DecisionVariableType> newSolution =
                Eigen::Map<Eigen::ArrayXd>(message.data() + i + 1, decisionVariablesNum);
            Eigen::ArrayXd newObjectives =
                Eigen::Map<Eigen::ArrayXd>(message.data() + i + 1 + decisionVariablesNum, objectivesNum);
            Individual<DecisionVariableType> newIndividual(std::move(newSolution), std::move(newObjectives));

            double newSubObjective = decomposition->ComputeObjective(individuals[index].weightVector, newIndividual.objectives);
            double oldSubObjective =
                decomposition->ComputeObjective(individuals[index].weightVector, individuals[index].objectives);
            if (newSubObjective < oldSubObjective) {
                individuals[index].UpdateFrom(newIndividual);
                if (isInternal) {
                    updatedSolutionIndexes.insert(index);
                } else {
                    updatedExternalIndexes.insert(index);
                }
            }

            UpdateIdealPoint(newIndividual.objectives);
        }

        if (containsIdealPoint) {
            UpdateIdealPointWithMessage(message);
        }
    }

#ifdef _TEST_
   public:
    OneNtMoead(int totalPopulationSize, int generationNum, int decisionVariableNum, int objectiveNum, int neighborNum,
               int migrationInterval, int H)
        : totalPopulationSize(totalPopulationSize),
          generationNum(generationNum),
          decisionVariablesNum(decisionVariableNum),
          objectivesNum(objectiveNum),
          neighborhoodSize(neighborNum),
          migrationInterval(migrationInterval),
          divisionsNumOfWeightVector(H) {}

    friend class OneNtMoeadTest;
    friend class OneNtMoeadTestM;
#endif
};

}  // namespace Eacpp
