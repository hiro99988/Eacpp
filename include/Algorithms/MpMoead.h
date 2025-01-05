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
class MpMoead : public IMoead<DecisionVariableType> {
   public:
    constexpr static int maxBufferSize = 100;
    constexpr static int messageTag = 0;

   private:
    int _totalPopulationSize;
    int _generationNum;
    int _currentGeneration;
    int _decisionVariablesNum;
    int _objectivesNum;
    int _neighborhoodSize;
    int _migrationInterval;
    int _divisionsNumOfWeightVector;
    std::shared_ptr<ICrossover<DecisionVariableType>> _crossover;
    std::shared_ptr<IDecomposition> _decomposition;
    std::shared_ptr<IMutation<DecisionVariableType>> _mutation;
    std::shared_ptr<IProblem<DecisionVariableType>> _problem;
    std::shared_ptr<IRepair<DecisionVariableType>> _repair;
    std::shared_ptr<ISampling<DecisionVariableType>> _sampling;
    std::shared_ptr<ISelection> _selection;
    int _rank;
    int _parallelSize;
    MoeadInitializer _initializer;
    std::vector<int> _internalIndexes;
    std::vector<int> _externalIndexes;
    std::set<int> _updatedSolutionIndexes;
    std::unordered_map<int, Individual<DecisionVariableType>> _individuals;
    std::unordered_map<int, Individual<DecisionVariableType>>
        _clonedExternalIndividuals;
    std::vector<int> _ranksForExternalIndividuals;
    std::set<int> _neighboringRanks;
    std::vector<int> _ranksToSend;
    int _singleMessageSize;
    bool _idealPointMigration;
    bool _isIdealPointUpdated;

   public:
    MpMoead(int generationNum, int neighborhoodSize,
            int divisionsNumOfWeightVector, int migrationInterval,
            const std::shared_ptr<ICrossover<DecisionVariableType>>& crossover,
            const std::shared_ptr<IDecomposition>& decomposition,
            const std::shared_ptr<IMutation<DecisionVariableType>>& mutation,
            const std::shared_ptr<IProblem<DecisionVariableType>>& problem,
            const std::shared_ptr<IRepair<DecisionVariableType>>& repair,
            const std::shared_ptr<ISampling<DecisionVariableType>>& sampling,
            const std::shared_ptr<ISelection>& selection,
            bool idealPointMigration = false)
        : _generationNum(generationNum),
          _neighborhoodSize(neighborhoodSize),
          _migrationInterval(migrationInterval),
          _divisionsNumOfWeightVector(divisionsNumOfWeightVector) {
        if (!crossover || !decomposition || !mutation || !problem || !repair ||
            !sampling || !selection) {
            throw std::invalid_argument("Null pointer is passed");
        }

        this->_crossover = crossover;
        this->_decomposition = decomposition;
        this->_mutation = mutation;
        this->_problem = problem;
        this->_repair = repair;
        this->_sampling = sampling;
        this->_selection = selection;
        this->_idealPointMigration = idealPointMigration;
        _decisionVariablesNum = problem->DecisionVariablesNum();
        _objectivesNum = problem->ObjectivesNum();
        _currentGeneration = 0;
        _singleMessageSize = _decisionVariablesNum + _objectivesNum + 1;
        _isIdealPointUpdated = false;
    }

    int CurrentGeneration() const override {
        return _currentGeneration;
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
        _totalPopulationSize = _initializer.CalculatePopulationSize(
            _divisionsNumOfWeightVector, _objectivesNum);
        _decomposition->InitializeIdealPoint(_objectivesNum);
        InitializeIsland();
        _currentGeneration = 0;
    }

    void Update() override {
        if (_currentGeneration % _migrationInterval == 0) {
            MakeLocalCopyOfExternalIndividuals();
        }

        for (auto&& i : _internalIndexes) {
            Individual<DecisionVariableType> newIndividual =
                GenerateNewIndividual(i);
            _repair->Repair(newIndividual);
            _problem->ComputeObjectiveSet(newIndividual);
            UpdateIdealPoint(newIndividual.objectives);
            UpdateNeighboringIndividuals(i, newIndividual);
        }

        _currentGeneration++;

        if (_currentGeneration % _migrationInterval == 0) {
            SendMessages();
            auto messages = ReceiveMessages();

            _updatedSolutionIndexes.clear();
            _isIdealPointUpdated = false;

            for (auto&& message : messages) {
                if (message.empty()) {
                    continue;
                }

                if (message.size() == _objectivesNum) {
                    UpdateIdealPointWithMessage(message);
                } else {
                    UpdateWithMessage(message);
                }
            }
        }
    }

    bool IsEnd() const override {
        return _currentGeneration >= _generationNum;
    }

    std::vector<Eigen::ArrayXd> GetObjectivesList() const override {
        std::vector<Eigen::ArrayXd> objectives;
        for (auto&& i : _internalIndexes) {
            objectives.push_back(_individuals.at(i).objectives);
        }

        return objectives;
    }

    std::vector<Eigen::ArrayX<DecisionVariableType>> GetSolutionList()
        const override {
        std::vector<Eigen::ArrayX<DecisionVariableType>> solutions;
        for (auto&& i : _internalIndexes) {
            solutions.push_back(_individuals.at(i).solution);
        }

        return solutions;
    }

   private:
    void Clear() {
        _internalIndexes.clear();
        _externalIndexes.clear();
        _updatedSolutionIndexes.clear();
        _individuals.clear();
        _clonedExternalIndividuals.clear();
        _ranksForExternalIndividuals.clear();
        _neighboringRanks.clear();
        _ranksToSend.clear();
    }

    void InitializeMpi() {
        int initialized;
        MPI_Initialized(&initialized);
        if (!initialized) {
            MPI_Init(nullptr, nullptr);
        }
        MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &_parallelSize);
    }

    void InitializeIsland() {
        std::vector<double> weightVectors1d;
        std::vector<int> neighborhoodIndexes1d;
        std::vector<int> populationSizes;
        if (_rank == 0) {
            _initializer.GenerateWeightVectorsAndNeighborhoods(
                _divisionsNumOfWeightVector, _objectivesNum, _neighborhoodSize,
                weightVectors1d, neighborhoodIndexes1d);
            populationSizes =
                CalculateNodeWorkloads(_totalPopulationSize, _parallelSize);
        }

        // 重みベクトルと近傍インデックスを分散
        std::vector<double> receivedWeightVectors =
            Scatterv(weightVectors1d, populationSizes, _objectivesNum, _rank,
                     _parallelSize);
        std::vector<int> receivedNeighborhoodIndexes =
            Scatterv(neighborhoodIndexes1d, populationSizes, _neighborhoodSize,
                     _rank, _parallelSize);

        // 各ノードの近傍のインデックスと重みベクトルを生成
        std::vector<int> noduplicateNeighborhoodIndexes;
        std::vector<int> neighborhoodSizes;
        std::vector<double> sendExternalNeighboringWeightVectors;
        if (_rank == 0) {
            std::tie(noduplicateNeighborhoodIndexes, neighborhoodSizes) =
                GenerateExternalNeighborhood(neighborhoodIndexes1d,
                                             populationSizes);
            sendExternalNeighboringWeightVectors =
                GetWeightVectorsMatchingIndexes(weightVectors1d,
                                                noduplicateNeighborhoodIndexes);
        }

        // 各ノードの近傍のインデックスと重みベクトルを分散
        _externalIndexes = Scatterv(noduplicateNeighborhoodIndexes,
                                    neighborhoodSizes, 1, _rank, _parallelSize);
        std::vector<double> receivedExternalNeighboringWeightVectors =
            Scatterv(sendExternalNeighboringWeightVectors, neighborhoodSizes,
                     _objectivesNum, _rank, _parallelSize);

        // 通信対象のノードと送信するインデックスを分散
        std::vector<int> sendRanksToSentByRank;
        std::vector<int> ranksToSentByRankSizes;
        if (_rank == 0) {
            CalculateRanksToSent(neighborhoodIndexes1d, populationSizes,
                                 sendRanksToSentByRank, ranksToSentByRankSizes);
        }
        _ranksToSend = Scatterv(sendRanksToSentByRank, ranksToSentByRankSizes,
                                1, _rank, _parallelSize);

        // 受信したデータを2Dに変換
        std::vector<Eigen::ArrayXd> weightVectors =
            TransformToEigenArrayX2d(receivedWeightVectors, _objectivesNum);
        std::vector<std::vector<int>> neighborhoodIndexes =
            TransformTo2d(receivedNeighborhoodIndexes, _neighborhoodSize);
        std::vector<Eigen::ArrayXd> externalNeighboringWeightVectors =
            TransformToEigenArrayX2d(receivedExternalNeighboringWeightVectors,
                                     _objectivesNum);

        _internalIndexes =
            GenerateNodeIndexes(_totalPopulationSize, _rank, _parallelSize);
        CalculateNeighboringRanks();
        InitializePopulation();

        auto receivedExternalSolutions = ScatterPopulation();

        InitializeExternalPopulation(receivedExternalSolutions);
        InitializeIndividualAndWeightVector(weightVectors, neighborhoodIndexes,
                                            externalNeighboringWeightVectors);
    }

    std::pair<std::vector<int>, std::vector<int>> GenerateExternalNeighborhood(
        std::vector<int>& neighborhoodIndexes,
        std::vector<int>& populationSizes) {
        std::vector<int> noduplicateNeighborhoodIndexes;
        std::vector<int> neighborhoodSizes;
        for (int i = 0; i < populationSizes.size(); i++) {
            int start = std::reduce(populationSizes.begin(),
                                    populationSizes.begin() + i);
            int end = start + populationSizes[i];

            std::vector<int> indexes(
                neighborhoodIndexes.begin() + (start * _neighborhoodSize),
                neighborhoodIndexes.begin() + (end * _neighborhoodSize));

            // 重複を削除
            std::sort(indexes.begin(), indexes.end());
            indexes.erase(std::unique(indexes.begin(), indexes.end()),
                          indexes.end());

            // 自分の担当する解のインデックスを削除
            std::erase_if(indexes, [&](int index) {
                return start <= index && index < end;
            });

            noduplicateNeighborhoodIndexes.insert(
                noduplicateNeighborhoodIndexes.end(), indexes.begin(),
                indexes.end());
            neighborhoodSizes.push_back(indexes.size());
        }
        return {noduplicateNeighborhoodIndexes, neighborhoodSizes};
    }

    std::vector<double> GetWeightVectorsMatchingIndexes(
        std::vector<double>& weightVectors, std::vector<int>& indexes) {
        std::vector<double> matchingWeightVectors;
        for (int i = 0; i < indexes.size(); i++) {
            matchingWeightVectors.insert(
                matchingWeightVectors.end(),
                weightVectors.begin() + indexes[i] * _objectivesNum,
                weightVectors.begin() + (indexes[i] + 1) * _objectivesNum);
        }
        return matchingWeightVectors;
    }

    void CalculateNeighboringRanks() {
        _ranksForExternalIndividuals.reserve(_externalIndexes.size());
        for (auto&& i : _externalIndexes) {
            int rank = GetRankFromIndex(_totalPopulationSize, i, _parallelSize);
            _ranksForExternalIndividuals.push_back(rank);
            _neighboringRanks.insert(rank);
        }
    }

    void InitializeIndividualAndWeightVector(
        std::vector<Eigen::ArrayXd>& weightVectors,
        std::vector<std::vector<int>>& neighborhoodIndexes,
        std::vector<Eigen::ArrayXd>& externalNeighboringWeightVectors) {
        for (int i = 0; i < _internalIndexes.size(); i++) {
            _individuals[_internalIndexes[i]].weightVector =
                std::move(weightVectors[i]);
            _individuals[_internalIndexes[i]].neighborhood =
                std::move(neighborhoodIndexes[i]);
        }
        for (int i = 0; i < _externalIndexes.size(); i++) {
            _clonedExternalIndividuals[_externalIndexes[i]].weightVector =
                std::move(externalNeighboringWeightVectors[i]);
        }
    }

    void CalculateRanksToSent(const std::vector<int>& neighborhoodIndexes,
                              const std::vector<int>& populationSizes,
                              std::vector<int>& outRanksToSentByRank,
                              std::vector<int>& outSizes) {
        std::vector<std::set<int>> ranksToSentByRank(_parallelSize,
                                                     std::set<int>());
        for (int dest = 0, count = 0; dest < _parallelSize;
             count += populationSizes[dest] * _neighborhoodSize, ++dest) {
            std::vector<int> neighborhood;
            std::copy(neighborhoodIndexes.begin() + count,
                      neighborhoodIndexes.begin() + count +
                          populationSizes[dest] * _neighborhoodSize,
                      std::back_inserter(neighborhood));
            std::sort(neighborhood.begin(), neighborhood.end());
            neighborhood.erase(
                std::unique(neighborhood.begin(), neighborhood.end()),
                neighborhood.end());

            for (auto&& i : neighborhood) {
                int source =
                    GetRankFromIndex(_totalPopulationSize, i, _parallelSize);
                if (source != dest) {
                    ranksToSentByRank[source].insert(dest);
                }
            }
        }

        for (auto&& i : ranksToSentByRank) {
            outSizes.push_back(i.size());
        }

        outRanksToSentByRank.reserve(_parallelSize * _neighborhoodSize);
        for (auto&& i : ranksToSentByRank) {
            for (auto&& j : i) {
                outRanksToSentByRank.push_back(j);
            }
        }
    }

    std::vector<std::vector<double>> ScatterPopulation() {
        std::vector<double> individualsToSend;
        individualsToSend.reserve(
            _idealPointMigration
                ? _internalIndexes.size() * _singleMessageSize + _objectivesNum
                : _internalIndexes.size() * _singleMessageSize);
        for (auto&& i : _internalIndexes) {
            individualsToSend.push_back(i);
            individualsToSend.insert(individualsToSend.end(),
                                     _individuals[i].solution.begin(),
                                     _individuals[i].solution.end());
            individualsToSend.insert(individualsToSend.end(),
                                     _individuals[i].objectives.begin(),
                                     _individuals[i].objectives.end());
        }

        if (_idealPointMigration) {
            individualsToSend.insert(individualsToSend.end(),
                                     _decomposition->IdealPoint().begin(),
                                     _decomposition->IdealPoint().end());
        }

        std::vector<MPI_Request> requests;
        requests.reserve(_ranksToSend.size());
        for (auto&& i : _ranksToSend) {
            requests.emplace_back();
            MPI_Isend(individualsToSend.data(), individualsToSend.size(),
                      MPI_DOUBLE, i, messageTag, MPI_COMM_WORLD,
                      &requests.back());
        }

        std::vector<std::vector<double>> receivedIndividuals;
        receivedIndividuals.reserve(_neighboringRanks.size());
        for (auto&& rank : _neighboringRanks) {
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

    void InitializePopulation() {
        int sampleNum = _internalIndexes.size();
        std::vector<Individual<DecisionVariableType>> sampledIndividuals =
            _sampling->Sample(sampleNum, _decisionVariablesNum);
        for (int i = 0; i < _internalIndexes.size(); i++) {
            _individuals[_internalIndexes[i]] = sampledIndividuals[i];
            _problem->ComputeObjectiveSet(_individuals[_internalIndexes[i]]);
            _decomposition->UpdateIdealPoint(
                _individuals[_internalIndexes[i]].objectives);
        }
    }

    void InitializeExternalPopulation(
        std::vector<std::vector<double>>& receivedIndividuals) {
        for (auto&& receive : receivedIndividuals) {
            int limit = _idealPointMigration ? receive.size() - _objectivesNum
                                             : receive.size();
            for (int i = 0; i < limit; i += _singleMessageSize) {
                int index = receive[i];
                if (!IsExternal(index)) {
                    continue;
                }

                _clonedExternalIndividuals[index].solution =
                    Eigen::Map<Eigen::ArrayXd>(receive.data() + (i + 1),
                                               _decisionVariablesNum);
                _clonedExternalIndividuals[index].objectives =
                    Eigen::Map<Eigen::ArrayXd>(
                        receive.data() + (i + 1 + _decisionVariablesNum),
                        _objectivesNum);
                UpdateIdealPoint(_clonedExternalIndividuals[index].objectives);
            }

            if (_idealPointMigration) {
                UpdateIdealPointWithMessage(receive);
            }
        }
    }

    void MakeLocalCopyOfExternalIndividuals() {
        for (auto&& i : _externalIndexes) {
            _individuals[i] = _clonedExternalIndividuals[i];
        }
    }

    std::vector<Individual<DecisionVariableType>> SelectParents(int index) {
        std::vector<int> parentCandidates;
        std::copy_if(_individuals[index].neighborhood.begin(),
                     _individuals[index].neighborhood.end(),
                     std::back_inserter(parentCandidates),
                     [index](int i) { return i != index; });
        std::vector<int> parentIndexes = _selection->Select(
            _crossover->GetParentNum() - 1, parentCandidates);
        std::vector<Individual<DecisionVariableType>> parents;
        parents.push_back(_individuals[index]);
        for (auto&& i : parentIndexes) {
            parents.push_back(_individuals[i]);
        }
        return parents;
    }

    Individual<DecisionVariableType> GenerateNewIndividual(int index) {
        std::vector<Individual<DecisionVariableType>> parents =
            SelectParents(index);
        Individual<DecisionVariableType> newIndividual =
            _crossover->Cross(parents);
        _mutation->Mutate(newIndividual);
        return newIndividual;
    }

    void UpdateNeighboringIndividuals(
        int index, Individual<DecisionVariableType>& newIndividual) {
        for (auto&& i : _individuals[index].neighborhood) {
            double newSubObjective = _decomposition->ComputeObjective(
                _individuals[i].weightVector, newIndividual.objectives);
            double oldSubObjective = _decomposition->ComputeObjective(
                _individuals[i].weightVector, _individuals[i].objectives);
            if (newSubObjective < oldSubObjective) {
                _individuals[i].UpdateFrom(newIndividual);
                if (IsInternal(i)) {
                    _updatedSolutionIndexes.insert(i);
                }
            }
        }
    }

    bool IsInternal(int index) {
        return std::find(_internalIndexes.begin(), _internalIndexes.end(),
                         index) != _internalIndexes.end();
    }

    bool IsExternal(int index) {
        return std::find(_externalIndexes.begin(), _externalIndexes.end(),
                         index) != _externalIndexes.end();
    }

    void UpdateIdealPoint(const Eigen::ArrayXd& objectives) {
        auto idealPoint = _decomposition->IdealPoint();
        for (int i = 0; i < _objectivesNum; i++) {
            if (objectives(i) < idealPoint(i)) {
                _isIdealPointUpdated = true;
                _decomposition->UpdateIdealPoint(objectives);
                break;
            }
        }
    }

    void UpdateIdealPointWithMessage(const std::vector<double>& message) {
        if (message.size() % _singleMessageSize == _objectivesNum) {
            Eigen::ArrayXd receivedIdealPoint =
                Eigen::Map<const Eigen::ArrayXd>(
                    message.data() + (message.size() - _objectivesNum),
                    _objectivesNum);
            UpdateIdealPoint(receivedIdealPoint);
        }
    }

    std::unordered_map<int, std::vector<double>> CreateMessages() {
        std::vector<double> updatedInternalIndividuals;
        updatedInternalIndividuals.reserve(_updatedSolutionIndexes.size() *
                                           _singleMessageSize);
        for (auto&& i : _updatedSolutionIndexes) {
            updatedInternalIndividuals.push_back(i);
            updatedInternalIndividuals.insert(updatedInternalIndividuals.end(),
                                              _individuals[i].solution.begin(),
                                              _individuals[i].solution.end());
            updatedInternalIndividuals.insert(
                updatedInternalIndividuals.end(),
                _individuals[i].objectives.begin(),
                _individuals[i].objectives.end());
        }

        std::unordered_map<int, std::vector<double>> dataToSend;
        for (int i = 0; i < _externalIndexes.size(); i++) {
            int index = _externalIndexes[i];
            bool updated = (_individuals[index].solution !=
                            _clonedExternalIndividuals[index].solution)
                               .any();
            if (updated) {
                int rank = _ranksForExternalIndividuals[i];
                dataToSend[rank].push_back(index);
                dataToSend[rank].insert(dataToSend[rank].end(),
                                        _individuals[index].solution.begin(),
                                        _individuals[index].solution.end());
                dataToSend[rank].insert(dataToSend[rank].end(),
                                        _individuals[index].objectives.begin(),
                                        _individuals[index].objectives.end());
            }
        }

        for (auto&& rank : _neighboringRanks) {
            dataToSend[rank].insert(dataToSend[rank].end(),
                                    updatedInternalIndividuals.begin(),
                                    updatedInternalIndividuals.end());
            if (_idealPointMigration && _isIdealPointUpdated) {
                dataToSend[rank].insert(dataToSend[rank].end(),
                                        _decomposition->IdealPoint().begin(),
                                        _decomposition->IdealPoint().end());
            }
        }

        return dataToSend;
    }

    void SendMessages() {
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
            MPI_Isend(message.data(), message.size(), MPI_DOUBLE, dest,
                      messageTag, MPI_COMM_WORLD, &request);
        }
    }

    std::vector<std::vector<double>> ReceiveMessages() {
        std::vector<std::vector<double>> receiveMessages;
        for (auto&& source : _neighboringRanks) {
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

    void UpdateWithMessage(std::vector<double>& message) {
        int limit = message.size();
        bool containsIdealPoint =
            _idealPointMigration &&
            message.size() % _singleMessageSize == _objectivesNum;
        if (containsIdealPoint) {
            limit -= _objectivesNum;
        }
        for (int i = 0; i < limit; i += _singleMessageSize) {
            int index = message[i];
            if (!(IsInternal(index) || IsExternal(index))) {
                continue;
            }

            Eigen::ArrayX<DecisionVariableType> newSolution =
                Eigen::Map<Eigen::ArrayXd>(message.data() + i + 1,
                                           _decisionVariablesNum);
            Eigen::ArrayXd newObjectives = Eigen::Map<Eigen::ArrayXd>(
                message.data() + i + 1 + _decisionVariablesNum, _objectivesNum);
            Individual<DecisionVariableType> newIndividual(
                std::move(newSolution), std::move(newObjectives));
            if (IsExternal(index)) {
                _clonedExternalIndividuals[index].UpdateFrom(newIndividual);
            } else {
                double newSubObjective = _decomposition->ComputeObjective(
                    _individuals[index].weightVector, newIndividual.objectives);
                double oldSubObjective = _decomposition->ComputeObjective(
                    _individuals[index].weightVector,
                    _individuals[index].objectives);
                if (newSubObjective < oldSubObjective) {
                    _individuals[index].UpdateFrom(newIndividual);
                    _updatedSolutionIndexes.insert(index);
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
    MpMoead(int totalPopulationSize, int generationNum, int decisionVariableNum,
            int objectiveNum, int neighborNum, int migrationInterval, int H)
        : _totalPopulationSize(totalPopulationSize),
          generationNum(generationNum),
          decisionVariablesNum(decisionVariableNum),
          _objectivesNum(objectiveNum),
          neighborhoodSize(neighborNum),
          migrationInterval(migrationInterval),
          divisionsNumOfWeightVector(H) {}

    friend class MpMoeadTest;
    friend class MpMoeadTestM;
#endif
};

}  // namespace Eacpp
