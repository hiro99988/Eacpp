#pragma once

#include <mpi.h>

#include <algorithm>
#include <array>
#include <eigen3/Eigen/Core>
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
class NewMoead : public IMoead<DecisionVariableType> {
   protected:
    constexpr static int maxBufferSize = 100;
    constexpr static int messageTag = 0;
    constexpr static int IndexTag = 0;
    constexpr static int SolutionTag = 1;
    constexpr static int ObjectiveTag = 2;
    constexpr static int IdealPointTag = 3;

    int totalPopulationSize;
    int generationNum;
    int currentGeneration;
    int decisionVariablesNum;
    int objectivesNum;
    int neighborhoodSize;
    int migrationInterval;
    int divisionsNumOfWeightVector;
    bool isIdealPointMigration;
    int singleMessageSize;
    std::shared_ptr<ICrossover<DecisionVariableType>> crossover;
    std::shared_ptr<IDecomposition> decomposition;
    std::shared_ptr<IMutation<DecisionVariableType>> mutation;
    std::shared_ptr<IProblem<DecisionVariableType>> problem;
    std::shared_ptr<IRepair<DecisionVariableType>> repair;
    std::shared_ptr<ISampling<DecisionVariableType>> sampling;
    std::shared_ptr<ISelection> selection;
    MoeadInitializer initializer;
    int rank;
    int parallelSize;
    bool isIdealPointUpdated;
    std::vector<int> internalIndexes;
    std::vector<int> externalIndexes;
    std::set<int> updatedSolutionIndexes;
    std::unordered_map<int, Individual<DecisionVariableType>> individuals;
    std::vector<int> destinationRanks;

   public:
    NewMoead(int generationNum, int neighborhoodSize,
             int divisionsNumOfWeightVector, int migrationInterval,
             bool isIdealPointMigration,
             const std::shared_ptr<ICrossover<DecisionVariableType>>& crossover,
             const std::shared_ptr<IDecomposition>& decomposition,
             const std::shared_ptr<IMutation<DecisionVariableType>>& mutation,
             const std::shared_ptr<IProblem<DecisionVariableType>>& problem,
             const std::shared_ptr<IRepair<DecisionVariableType>>& repair,
             const std::shared_ptr<ISampling<DecisionVariableType>>& sampling,
             const std::shared_ptr<ISelection>& selection)
        : generationNum(generationNum),
          neighborhoodSize(neighborhoodSize),
          divisionsNumOfWeightVector(divisionsNumOfWeightVector),
          migrationInterval(migrationInterval),
          isIdealPointMigration(isIdealPointMigration) {
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
    virtual ~NewMoead() {}

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
        totalPopulationSize = initializer.CalculatePopulationSize(
            divisionsNumOfWeightVector, objectivesNum);
        decomposition->InitializeIdealPoint(objectivesNum);
        InitializeIsland();
        currentGeneration = 0;
    }

    void Update() override {
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
                UpdateWithMessage(message);
            }
        }
    }

    bool IsEnd() const override {
        return currentGeneration >= generationNum;
    }

    std::vector<Eigen::ArrayXd> GetObjectivesList() const override {
        std::vector<Eigen::ArrayXd> objectives;
        objectives.reserve(internalIndexes.size());
        for (auto&& i : internalIndexes) {
            objectives.push_back(individuals.at(i).objectives);
        }

        return objectives;
    }
    std::vector<Eigen::ArrayX<DecisionVariableType>> GetSolutionList()
        const override {
        std::vector<Eigen::ArrayX<DecisionVariableType>> solutions;
        solutions.reserve(internalIndexes.size());
        for (auto&& i : internalIndexes) {
            solutions.push_back(individuals.at(i).solution);
        }

        return solutions;
    }

   protected:
    void InitializeMpi() {
        int initialized;
        MPI_Initialized(&initialized);
        if (!initialized) {
            MPI_Init(nullptr, nullptr);
        }

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &parallelSize);
    }

    void InitializeIsland();
    void Clear();
    std::pair<std::vector<int>, std::vector<int>> GenerateExternalNeighborhood(
        std::vector<int>& neighborhoodIndexes,
        std::vector<int>& populationSizes);

    std::vector<double> GetWeightVectorsMatchingIndexes(
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

    std::set<int> CalculateNeighboringRanks();

    void CalculateDestinationRanks(const std::vector<int>& neighborhoodIndexes,
                                   const std::vector<int>& populationSizes,
                                   std::vector<int>& outDestinationRanksByRank,
                                   std::vector<int>& outSizes) {
        std::vector<std::set<int>> destinationRanksByRank(parallelSize,
                                                          std::set<int>());
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
                int source =
                    GetRankFromIndex(totalPopulationSize, i, parallelSize);
                if (source != dest) {
                    destinationRanksByRank[source].insert(dest);
                }
            }
        }

        for (auto&& i : destinationRanksByRank) {
            outSizes.push_back(i.size());
        }

        outDestinationRanksByRank.reserve(parallelSize * neighborhoodSize);
        for (auto&& i : destinationRanksByRank) {
            for (auto&& j : i) {
                outDestinationRanksByRank.push_back(j);
            }
        }
    }

    std::vector<std::vector<double>> ScatterPopulation(
        const std::vector<int>& destinationRanks,
        const std::set<int>& recipients) {
        std::vector<double> individualsToSend;
        individualsToSend.reserve(
            isIdealPointMigration
                ? internalIndexes.size() * singleMessageSize + objectivesNum
                : internalIndexes.size() * singleMessageSize);

        // 自身の個体を入れる
        for (auto&& i : internalIndexes) {
            individualsToSend.push_back(i);
            individualsToSend.insert(individualsToSend.end(),
                                     individuals[i].solution.begin(),
                                     individuals[i].solution.end());
            individualsToSend.insert(individualsToSend.end(),
                                     individuals[i].objectives.begin(),
                                     individuals[i].objectives.end());
        }

        // ideal pointを入れる
        if (isIdealPointMigration) {
            individualsToSend.insert(individualsToSend.end(),
                                     decomposition->IdealPoint().begin(),
                                     decomposition->IdealPoint().end());
        }

        // 送信
        std::vector<MPI_Request> requests(destinationRanks.size());
        for (std::size_t i = 0; i < destinationRanks.size(); ++i) {
            MPI_Isend(individualsToSend.data(), individualsToSend.size(),
                      MPI_DOUBLE, destinationRanks[i], messageTag,
                      MPI_COMM_WORLD, &requests[i]);
        }

        // 受信
        std::vector<std::vector<double>> receivedIndividuals;
        receivedIndividuals.reserve(recipients.size());
        MPI_Status status;
        for (auto&& rank : recipients) {
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

    void InitializePopulation() {
        int sampleNum = internalIndexes.size();
        auto sampledIndividuals =
            sampling->Sample(sampleNum, decisionVariablesNum);
        for (int i = 0; i < internalIndexes.size(); i++) {
            int index = internalIndexes[i];
            individuals[index] = sampledIndividuals[i];
            problem->ComputeObjectiveSet(individuals[index]);
            decomposition->UpdateIdealPoint(individuals[index].objectives);
        }
    }

    void InitializeExternalIndividuals(
        const std::vector<std::vector<double>>& receivedIndividuals,
        std::unordered_map<int, Individual<DecisionVariableType>>&
            inputIndividuals) {
        for (auto&& receive : receivedIndividuals) {
            int limit = isIdealPointMigration ? receive.size() - objectivesNum
                                              : receive.size();
            for (int i = 0; i < limit; i += singleMessageSize) {
                int index = receive[i];
                if (!IsExternal(index)) {
                    continue;
                }

                inputIndividuals[index].solution = Eigen::Map<Eigen::ArrayXd>(
                    receive.data() + (i + 1), decisionVariablesNum);
                inputIndividuals[index].objectives = Eigen::Map<Eigen::ArrayXd>(
                    receive.data() + (i + 1 + decisionVariablesNum),
                    objectivesNum);
                UpdateIdealPoint(inputIndividuals[index].objectives);
            }

            if (isIdealPointMigration) {
                UpdateIdealPointWithMessage(receive);
            }
        }
    }

    void InitializeIndividualAndWeightVector(
        std::vector<Eigen::ArrayXd>& weightVectors,
        std::vector<std::vector<int>>& neighborhoodIndexes,
        std::vector<Eigen::ArrayXd>& externalWeightVectors) {
        for (int i = 0; i < internalIndexes.size(); i++) {
            individuals[internalIndexes[i]].weightVector =
                std::move(weightVectors[i]);
            individuals[internalIndexes[i]].neighborhood =
                std::move(neighborhoodIndexes[i]);
        }

        for (int i = 0; i < externalIndexes.size(); i++) {
            individuals[externalIndexes[i]].weightVector =
                std::move(externalWeightVectors[i]);
        }
    }

    std::vector<Individual<DecisionVariableType>&> SelectParents(int index) {
        std::vector<int> parentCandidates;
        parentCandidates.reserve(individuals[index].neighborhood.size() - 1);
        std::copy_if(individuals[index].neighborhood.begin(),
                     individuals[index].neighborhood.end(),
                     std::back_inserter(parentCandidates),
                     [index](int i) { return i != index; });

        std::vector<int> parentIndexes =
            selection->Select(crossover->GetParentNum() - 1, parentCandidates);

        std::vector<Individual<DecisionVariableType>&> parents;
        parents.reserve(crossover->GetParentNum());
        parents.push_back(individuals[index]);
        for (auto&& i : parentIndexes) {
            parents.push_back(individuals[i]);
        }

        return parents;
    }

    Individual<DecisionVariableType> GenerateNewIndividual(int index) {
        auto parents = SelectParents(index);
        Individual<DecisionVariableType> newIndividual =
            crossover->Cross(parents);
        mutation->Mutate(newIndividual);
        return newIndividual;
    }

    void UpdateNeighboringIndividuals(
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

    bool IsInternal(int index) {
        return std::find(internalIndexes.begin(), internalIndexes.end(),
                         index) != internalIndexes.end();
    }

    bool IsExternal(int index) {
        return std::find(externalIndexes.begin(), externalIndexes.end(),
                         index) != externalIndexes.end();
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
                Eigen::Map<const Eigen::ArrayXd>(
                    message.data() + (message.size() - objectivesNum),
                    objectivesNum);
            UpdateIdealPoint(receivedIdealPoint);
        }
    }

    std::unordered_map<int, std::vector<double>> CreateMessages();

    void SendMessages() {
        // MPI_Isendで使うバッファ
        static std::array<std::unordered_map<int, std::vector<double>>,
                          maxBufferSize>
            messageToSendBuffers;
        static int messageToSendBufferIndex = 0;
        messageToSendBufferIndex =
            (messageToSendBufferIndex + 1) % maxBufferSize;

        std::unordered_map<int, std::vector<double>>& messagesToSend =
            messageToSendBuffers[messageToSendBufferIndex];
        messagesToSend = CreateMessages();

        // メッセージを送信する
        MPI_Request request;
        for (const auto& [dest, message] : messagesToSend) {
            MPI_Isend(message.data(), message.size(), MPI_DOUBLE, dest,
                      messageTag, MPI_COMM_WORLD, &request);
        }
    }

    std::vector<std::vector<double>> ReceiveMessages() {
        std::vector<std::vector<double>> receivedMessages;
        receivedMessages.reserve(neighboringRanks.size() * migrationInterval);
        for (auto&& source : neighboringRanks) {
            IprobeAndRecvLoop(source, receivedMessages);
        }

        return receivedMessages;
    }

    void IprobeAndRecvLoop(
        int source, std::vector<std::vector<double>>& outReceivedMessages) {
        while (true) {
            MPI_Status status;
            int canReceive;
            MPI_Iprobe(source, messageTag, MPI_COMM_WORLD, &canReceive,
                       &status);
            if (!canReceive) {
                break;
            }

            int receivedDataSize;
            MPI_Get_count(&status, MPI_DOUBLE, &receivedDataSize);
            std::vector<double> receive(receivedDataSize);
            MPI_Recv(receive.data(), receivedDataSize, MPI_DOUBLE, source,
                     messageTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            outReceivedMessages.push_back(std::move(receive));
        }
    }

    void UpdateWithMessage(std::vector<double>& message);

#ifdef _TEST_
   public:
    friend class MpMoeadTest;
    friend class MpMoeadTestM;
#endif
};

}  // namespace Eacpp