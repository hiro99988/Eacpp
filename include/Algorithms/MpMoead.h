#pragma once

#include <mpi.h>

#include <algorithm>
#include <list>
#include <memory>
#include <numeric>
#include <ranges>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Algorithms/IParallelMoead.hpp"
#include "Algorithms/MoeadInitializer.h"
#include "Crossovers/ICrossover.h"
#include "Decompositions/IDecomposition.h"
#include "Individual.h"
#include "Mutations/IMutation.h"
#include "Problems/IProblem.h"
#include "Repairs/IRepair.h"
#include "Samplings/ISampling.h"
#include "Selections/ISelection.h"
#include "Stopwatches/MpiStopwatch.hpp"
#include "Utils/MpiUtils.h"
#include "Utils/Utils.h"

namespace Eacpp {

template <typename DecisionVariableType>
class MpMoead : public IParallelMoead<DecisionVariableType> {
   protected:
    // 送信中のデータを管理する構造体
    struct PendingSend {
        MPI_Request request;
        std::vector<double> data;
    };

    constexpr static int MessageTag = 0;

    int _generationNum;
    int _neighborhoodSize;
    int _divisionsNumOfWeightVector;
    int _migrationInterval;
    bool _isAsync;
    int _decisionVariablesNum;
    int _objectivesNum;
    int _singleMessageSize;
    std::shared_ptr<ICrossover<DecisionVariableType>> _crossover;
    std::shared_ptr<IDecomposition> _decomposition;
    std::shared_ptr<IMutation<DecisionVariableType>> _mutation;
    std::shared_ptr<IProblem<DecisionVariableType>> _problem;
    std::shared_ptr<IRepair<DecisionVariableType>> _repair;
    std::shared_ptr<ISampling<DecisionVariableType>> _sampling;
    std::shared_ptr<ISelection> _selection;
    int _rank;
    int _parallelSize;
    int _totalPopulationSize;
    int _currentGeneration;
    bool _isIdealPointUpdated;
    MoeadInitializer _initializer;
    std::vector<int> _internalIndexes;
    std::vector<int> _externalIndexes;
    std::set<int> _updatedSolutionIndexes;
    std::unordered_map<int, Individual<DecisionVariableType>> _individuals;
    std::unordered_map<int, Individual<DecisionVariableType>>
        _clonedExternalIndividuals;
    std::vector<int> _externalIndividualRanks;
    std::vector<int> _neighboringRanks;
    std::vector<int> _ranksToSend;
    // 通信量を記録する
    std::vector<std::vector<int>> _traffics;
    // 保留中の送信データを管理するバッファ
    std::list<PendingSend> _pendingSends;
    MpiStopwatch _initializationTime;
    MpiStopwatch _executionTime;
    MpiStopwatch _communicationTime;

   public:
    MpMoead(int generationNum, int neighborhoodSize,
            int divisionsNumOfWeightVector, int migrationInterval,
            const std::shared_ptr<ICrossover<DecisionVariableType>>& crossover,
            const std::shared_ptr<IDecomposition>& decomposition,
            const std::shared_ptr<IMutation<DecisionVariableType>>& mutation,
            const std::shared_ptr<IProblem<DecisionVariableType>>& problem,
            const std::shared_ptr<IRepair<DecisionVariableType>>& repair,
            const std::shared_ptr<ISampling<DecisionVariableType>>& sampling,
            const std::shared_ptr<ISelection>& selection, bool isAsync = true)
        : _generationNum(generationNum),
          _neighborhoodSize(neighborhoodSize),
          _migrationInterval(migrationInterval),
          _divisionsNumOfWeightVector(divisionsNumOfWeightVector),
          _isAsync(isAsync) {
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
        _decisionVariablesNum = problem->DecisionVariablesNum();
        _objectivesNum = problem->ObjectivesNum();
        _currentGeneration = 0;
        _singleMessageSize = _decisionVariablesNum + _objectivesNum + 1;
        _isIdealPointUpdated = false;
    }

    virtual ~MpMoead() {}

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
        _initializationTime.Restart();
        Clear();
        InitializeMpi();
        _totalPopulationSize = _initializer.CalculatePopulationSize(
            _divisionsNumOfWeightVector, _objectivesNum);
        _decomposition->InitializeIdealPoint(_objectivesNum);
        InitializeIsland();
        AdditionalInitialization();
        _initializationTime.Stop();
    }

    void Update() override {
        _executionTime.Start();
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

            _communicationTime.Start();
            std::vector<std::vector<double>> messages;
            if (_isAsync) {
                messages = ReceiveMessagesAsync();
            } else {
                messages = ReceiveMessagesSync();
            }
            _communicationTime.Stop();

            // 受信データ量を記録する
            _executionTime.Stop();
            int receiveTimes = messages.size();
            int totalReceiveDataTraffic = 0;
            for (const auto& message : messages) {
                totalReceiveDataTraffic += message.size();
            }
            _traffics.back().push_back(receiveTimes);
            _traffics.back().push_back(totalReceiveDataTraffic);
            _executionTime.Start();

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

        _executionTime.Stop();

        if (IsEnd()) {
            Complete_pendingSends();
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

    double GetInitializationTime() const override {
        return _initializationTime.Elapsed();
    }

    double GetExecutionTime() const override {
        return _executionTime.Elapsed();
    }

    double GetCommunicationTime() const override {
        return _communicationTime.Elapsed();
    }

    std::vector<std::vector<int>> GetDataTraffics() const override {
        return _traffics;
    }

   protected:
    /// @brief 追加の初期化を行う
    virtual void AdditionalInitialization() {}

    /// @brief 追加の変数の初期化を行う
    virtual void AdditionalClear() {}

    /// @brief dataToSendに送信する理想点を追加する
    /// @param dataToSend
    virtual void PushIdealPointToSend(
        std::unordered_map<int, std::vector<double>>& dataToSend) {
        if (_isIdealPointUpdated) {
            for (auto&& rank : _ranksToSend) {
                dataToSend[rank].insert(dataToSend[rank].end(),
                                        _decomposition->IdealPoint().begin(),
                                        _decomposition->IdealPoint().end());
            }
        }
    }

   private:
    void Clear() {
        _currentGeneration = 0;
        _isIdealPointUpdated = false;
        _internalIndexes.clear();
        _externalIndexes.clear();
        _updatedSolutionIndexes.clear();
        _individuals.clear();
        _clonedExternalIndividuals.clear();
        _externalIndividualRanks.clear();
        _neighboringRanks.clear();
        _ranksToSend.clear();
        _traffics.clear();
        _pendingSends.clear();
        _executionTime.Reset();
        _communicationTime.Reset();
        AdditionalClear();
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
        std::vector<int> internalIndividualIndexes;
        std::vector<double> internalWeightVectors;
        std::vector<int> internalNeighborhoods;
        std::vector<int> internalIndividualCounts;
        std::vector<int> externalIndividualIndexes;
        std::vector<int> externalIndividualRanks;
        std::vector<double> externalWeightVectors;
        std::vector<int> externalIndividualCounts;
        std::vector<int> ranksToSendAtInitialization;
        std::vector<int> ranksToSendCounts;
        std::vector<int> neighboringRanks;
        std::vector<int> neighboringRankCounts;
        if (_rank == 0) {
            _initializer.InitializeParallelMoead(
                _divisionsNumOfWeightVector, _objectivesNum, _neighborhoodSize,
                _parallelSize, internalIndividualIndexes, internalWeightVectors,
                internalNeighborhoods, internalIndividualCounts,
                externalIndividualIndexes, externalIndividualRanks,
                externalWeightVectors, externalIndividualCounts,
                ranksToSendAtInitialization, ranksToSendCounts,
                neighboringRanks, neighboringRankCounts);
        }

        // 内部個体を分散する
        _internalIndexes =
            Scatterv(internalIndividualIndexes, internalIndividualCounts, 1,
                     _rank, _parallelSize);
        std::vector<double> receivedInternalWeightVectors =
            Scatterv(internalWeightVectors, internalIndividualCounts,
                     _objectivesNum, _rank, _parallelSize);
        std::vector<int> receivedInternalNeighborhoods =
            Scatterv(internalNeighborhoods, internalIndividualCounts,
                     _neighborhoodSize, _rank, _parallelSize);

        // 外部個体を分散する
        _externalIndexes =
            Scatterv(externalIndividualIndexes, externalIndividualCounts, 1,
                     _rank, _parallelSize);
        _externalIndividualRanks =
            Scatterv(externalIndividualRanks, externalIndividualCounts, 1,
                     _rank, _parallelSize);
        std::vector<double> receivedExternalWeightVectors =
            Scatterv(externalWeightVectors, externalIndividualCounts,
                     _objectivesNum, _rank, _parallelSize);

        // 初期化時に送信するノードを分散する
        _ranksToSend = Scatterv(ranksToSendAtInitialization, ranksToSendCounts,
                                1, _rank, _parallelSize);

        // 近傍のランクを分散する
        _neighboringRanks = Scatterv(neighboringRanks, neighboringRankCounts, 1,
                                     _rank, _parallelSize);

        // 受信したデータを2Dに変換する
        std::vector<Eigen::ArrayXd> weightVectors = TransformToEigenArrayX2d(
            receivedInternalWeightVectors, _objectivesNum);
        std::vector<std::vector<int>> neighborhoodIndexes =
            TransformTo2d(receivedInternalNeighborhoods, _neighborhoodSize);
        std::vector<Eigen::ArrayXd> externalNeighboringWeightVectors =
            TransformToEigenArrayX2d(receivedExternalWeightVectors,
                                     _objectivesNum);

        InitializePopulation(internalIndividualCounts);

        auto receivedExternalIndividuals = ScatterPopulation(_ranksToSend);

        InitializeExternalPopulation(receivedExternalIndividuals);
        InitializeIndividualAndWeightVector(weightVectors, neighborhoodIndexes,
                                            externalNeighboringWeightVectors);

        // 送信する必要のない外部個体のexternalIndividualRanksを-1にする
        for (auto&& i : _externalIndividualRanks) {
            if (std::ranges::find(_ranksToSend, i) == _ranksToSend.end()) {
                i = -1;
            }
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

    std::vector<std::vector<double>> ScatterPopulation(
        const std::vector<int>& ranksToSend) {
        std::vector<double> individualsToSend;
        individualsToSend.reserve(_internalIndexes.size() * _singleMessageSize +
                                  _objectivesNum);
        for (auto&& i : _internalIndexes) {
            individualsToSend.push_back(i);
            individualsToSend.insert(individualsToSend.end(),
                                     _individuals[i].solution.begin(),
                                     _individuals[i].solution.end());
            individualsToSend.insert(individualsToSend.end(),
                                     _individuals[i].objectives.begin(),
                                     _individuals[i].objectives.end());
        }

        individualsToSend.insert(individualsToSend.end(),
                                 _decomposition->IdealPoint().begin(),
                                 _decomposition->IdealPoint().end());

        std::vector<MPI_Request> requests;
        requests.reserve(ranksToSend.size());
        for (auto&& i : ranksToSend) {
            requests.emplace_back();
            MPI_Isend(individualsToSend.data(), individualsToSend.size(),
                      MPI_DOUBLE, i, MessageTag, MPI_COMM_WORLD,
                      &requests.back());
        }

        // 受信対象のランクを計算する
        std::vector<int> ranksToReceive = _externalIndividualRanks;
        RemoveDuplicates(ranksToReceive);

        std::vector<std::vector<double>> receivedIndividuals;
        receivedIndividuals.reserve(ranksToReceive.size());
        for (auto&& rank : ranksToReceive) {
            MPI_Status status;
            MPI_Probe(rank, MessageTag, MPI_COMM_WORLD, &status);
            int receivedDataSize;
            MPI_Get_count(&status, MPI_DOUBLE, &receivedDataSize);
            receivedIndividuals.emplace_back(receivedDataSize);
            MPI_Recv(receivedIndividuals.back().data(), receivedDataSize,
                     MPI_DOUBLE, rank, MessageTag, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }

        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

        return receivedIndividuals;
    }

    // 各ランクが初期個体をそれぞれでサンプルする関数
    // void InitializePopulation() {
    //     int sampleNum = _internalIndexes.size();
    //     std::vector<Individual<DecisionVariableType>> sampledIndividuals =
    //         _sampling->Sample(sampleNum, _decisionVariablesNum);
    //     for (int i = 0; i < _internalIndexes.size(); i++) {
    //         _individuals[_internalIndexes[i]] = sampledIndividuals[i];
    //         _problem->ComputeObjectiveSet(_individuals[_internalIndexes[i]]);
    //         _decomposition->UpdateIdealPoint(
    //             _individuals[_internalIndexes[i]].objectives);
    //     }
    // }

    void InitializePopulation(
        const std::vector<int>& internalIndividualCounts) {
        std::vector<DecisionVariableType> sendBuffer;
        if (_rank == 0) {
            int sampleNum = 0;
            for (auto&& i : internalIndividualCounts) {
                sampleNum += i;
            }

            std::vector<Individual<DecisionVariableType>> samples =
                _sampling->Sample(sampleNum, _decisionVariablesNum);

            sendBuffer.reserve(samples.size() * _decisionVariablesNum);
            // samplesの解を送信用に1次元に変換
            for (auto&& sample : samples) {
                sendBuffer.insert(sendBuffer.end(), sample.solution.begin(),
                                  sample.solution.end());
            }
        }

        // 解を分散
        auto received = Scatterv(sendBuffer, internalIndividualCounts,
                                 _decisionVariablesNum, _rank, _parallelSize);
        // 受信した解を2次元に変換
        std::vector<Eigen::ArrayX<DecisionVariableType>> receivedSolutions =
            TransformToEigenArrayX2d(received, _decisionVariablesNum);

        // 受信した解を個体に格納
        for (int i = 0; i < receivedSolutions.size(); i++) {
            _individuals.emplace(_internalIndexes[i], receivedSolutions[i]);
            _problem->ComputeObjectiveSet(_individuals[_internalIndexes[i]]);
            _decomposition->UpdateIdealPoint(
                _individuals[_internalIndexes[i]].objectives);
        }
    }

    void InitializeExternalPopulation(
        std::vector<std::vector<double>>& receivedIndividuals) {
        for (auto&& receive : receivedIndividuals) {
            int limit = receive.size() - _objectivesNum;
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

            UpdateIdealPointWithMessage(receive);
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
            if (_externalIndividualRanks[i] == -1) {
                continue;
            }

            int index = _externalIndexes[i];
            bool updated = (_individuals[index].solution !=
                            _clonedExternalIndividuals[index].solution)
                               .any();
            if (updated) {
                int rank = _externalIndividualRanks[i];
                dataToSend[rank].push_back(index);
                dataToSend[rank].insert(dataToSend[rank].end(),
                                        _individuals[index].solution.begin(),
                                        _individuals[index].solution.end());
                dataToSend[rank].insert(dataToSend[rank].end(),
                                        _individuals[index].objectives.begin(),
                                        _individuals[index].objectives.end());
            }
        }

        if ((!_isAsync) || (_isAsync && !updatedInternalIndividuals.empty())) {
            for (auto&& rank : _ranksToSend) {
                dataToSend[rank].insert(dataToSend[rank].end(),
                                        updatedInternalIndividuals.begin(),
                                        updatedInternalIndividuals.end());
            }
        }

        PushIdealPointToSend(dataToSend);

        return dataToSend;
    }

    void SendMessages() {
        auto sendMessages = CreateMessages();

        // 送信データ量を記録する
        _executionTime.Stop();
        int sendTimes = sendMessages.size();
        int totalSendDataTraffic = 0;
        for (const auto& [dest, message] : sendMessages) {
            totalSendDataTraffic += message.size();
        }
        _traffics.push_back(
            {_currentGeneration, sendTimes, totalSendDataTraffic});
        _executionTime.Start();

        _communicationTime.Start();
        // メッセージを送信する
        for (auto&& [dest, message] : sendMessages) {
            _pendingSends.emplace_back();
            auto& ps = _pendingSends.back();
            ps.data = std::move(message);
            MPI_Isend(ps.data.data(), ps.data.size(), MPI_DOUBLE, dest,
                      MessageTag, MPI_COMM_WORLD, &ps.request);
        }

        // 送信が完了したリクエストを削除する
        _pendingSends.remove_if([](PendingSend& ps) {
            int flag = 0;
            MPI_Test(&ps.request, &flag, MPI_STATUS_IGNORE);
            return flag;
        });
        _communicationTime.Stop();
    }

    std::vector<std::vector<double>> ReceiveMessagesSync() {
        std::vector<std::vector<double>> receiveMessages;
        receiveMessages.reserve(_neighboringRanks.size());

        for (const auto& source : _neighboringRanks) {
            MPI_Status status;
            MPI_Probe(source, MessageTag, MPI_COMM_WORLD, &status);
            int count;
            MPI_Get_count(&status, MPI_DOUBLE, &count);

            std::vector<double> message(count);
            MPI_Recv(message.data(), count, MPI_DOUBLE, source, MessageTag,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            receiveMessages.push_back(std::move(message));
        }

        return receiveMessages;
    }

    std::vector<std::vector<double>> ReceiveMessagesAsync() {
        std::vector<std::vector<double>> receiveMessages;
        for (auto&& source : _neighboringRanks) {
            while (true) {
                MPI_Status status;
                int canReceive;
                MPI_Iprobe(source, MessageTag, MPI_COMM_WORLD, &canReceive,
                           &status);
                if (!canReceive) {
                    break;
                }

                int receiveDataSize;
                MPI_Get_count(&status, MPI_DOUBLE, &receiveDataSize);
                std::vector<double> receive(receiveDataSize);
                MPI_Recv(receive.data(), receiveDataSize, MPI_DOUBLE, source,
                         MessageTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                receiveMessages.push_back(std::move(receive));
            }
        }

        return receiveMessages;
    }

    void UpdateWithMessage(std::vector<double>& message) {
        int limit = message.size();
        bool containsIdealPoint =
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

    void Complete_pendingSends() {
        MPI_Barrier(MPI_COMM_WORLD);
        ReleaseIsend(_parallelSize, MPI_DOUBLE);

        std::vector<MPI_Request> requests;
        for (auto&& ps : _pendingSends) {
            requests.push_back(ps.request);
        }

        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }
};

}  // namespace Eacpp
