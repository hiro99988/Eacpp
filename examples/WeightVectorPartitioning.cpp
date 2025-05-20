#include <iostream>
#include <numeric>
#include <random>

#include "Algorithms/MoeadInitializer.h"
#include "Algorithms/SimulatedAnnealing.hpp"
#include "Graph/SimpleGraph.hpp"
#include "Rng/Rng.h"
#include "Utils/Utils.h"

using namespace Eacpp;

using SolutionType = std::vector<std::vector<std::vector<int>>>;

void GenerateAdjacencyList(int divisionsNumOfWeightVector, int objectivesNum,
                           int neighborhoodSize, int parallelSize,
                           const MoeadInitializer& moeadInitializer,
                           const SolutionType& solution,
                           std::vector<int>& outRanksToSendAtInitialization,
                           std::vector<int>& outRanksToSendCounts,
                           std::vector<int>& outNeighboringRanks,
                           std::vector<int>& outNeighboringRankCounts) {
    std::vector<int> internalIndividualIndexes;
    std::vector<double> internalWeightVectors;
    std::vector<int> internalNeighborhoods;
    std::vector<int> internalIndividualCounts;
    std::vector<int> externalIndividualIndexes;
    std::vector<int> externalIndividualRanks;
    std::vector<double> externalWeightVectors;
    std::vector<int> externalIndividualCounts;
    moeadInitializer.InitializeParallelMoead(
        divisionsNumOfWeightVector, objectivesNum, neighborhoodSize,
        parallelSize, internalIndividualIndexes, internalWeightVectors,
        internalNeighborhoods, internalIndividualCounts,
        externalIndividualIndexes, externalIndividualRanks,
        externalWeightVectors, externalIndividualCounts,
        outRanksToSendAtInitialization, outRanksToSendCounts,
        outNeighboringRanks, outNeighboringRankCounts, solution);
}

SimpleGraph GenerateGraphFromAdjacencyList(
    const std::vector<int>& ranksToSendAtInitialization,
    const std::vector<int>& ranksToSendCounts,
    const std::vector<int>& neighboringRanks,
    const std::vector<int>& neighboringRankCounts) {
    SimpleGraph graph = SimpleGraph::EmptyGraph(ranksToSendCounts.size());
    // Add edges based on the ranks to which each node sends messages.
    int sendIndex = 0;
    for (int i = 0; i < ranksToSendCounts.size(); ++i) {
        for (int j = 0; j < ranksToSendCounts[i]; ++j) {
            int target = ranksToSendAtInitialization[sendIndex++];
            graph.AddEdge(i, target);
        }
    }

    // Add edges based on the neighboring reception information.
    int neighborIndex = 0;
    for (int i = 0; i < neighboringRankCounts.size(); ++i) {
        for (int j = 0; j < neighboringRankCounts[i]; ++j) {
            int neighbor = neighboringRanks[neighborIndex++];
            graph.AddEdge(neighbor, i);
        }
    }

    return graph;
}

SimpleGraph GenerateGraph(int divisionsNumOfWeightVector, int objectivesNum,
                          int neighborhoodSize, int parallelSize,
                          const MoeadInitializer& moeadInitializer,
                          const SolutionType& solution) {
    std::vector<int> ranksToSendAtInitialization;
    std::vector<int> ranksToSendCounts;
    std::vector<int> neighboringRanks;
    std::vector<int> neighboringRankCounts;
    GenerateAdjacencyList(
        divisionsNumOfWeightVector, objectivesNum, neighborhoodSize,
        parallelSize, moeadInitializer, solution, ranksToSendAtInitialization,
        ranksToSendCounts, neighboringRanks, neighboringRankCounts);
    return GenerateGraphFromAdjacencyList(ranksToSendAtInitialization,
                                          ranksToSendCounts, neighboringRanks,
                                          neighboringRankCounts);
}

double CalculateSquaredEuclideanDistance(const std::vector<int>& lhs,
                                         const std::vector<int>& rhs) {
    double sum = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        double diff = static_cast<double>(lhs[i]) - static_cast<double>(rhs[i]);
        sum += diff * diff;
    }
    return sum;
}

std::vector<std::vector<double>> ComputeDistanceMatrix(
    const SolutionType& solution) {
    size_t n = solution.size();
    std::vector<std::vector<double>> avgDists(n, std::vector<double>(n, 0.0));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            size_t count = 0;
            for (const auto& vecA : solution[i]) {
                for (const auto& vecB : solution[j]) {
                    sum += CalculateSquaredEuclideanDistance(vecA, vecB);
                    ++count;
                }
            }
            if (count > 0) {
                avgDists[i][j] = sum / static_cast<double>(count);
            }
        }
    }

    return avgDists;
}

std::size_t SelectIndexByWeightedRandom(const std::vector<double>& vec) {
    // 小さい値ほど選ばれやすくするため、重みを逆数で計算
    std::vector<double> invWeights(vec.size());
    double totalWeight = 0.0;
    const double epsilon = 1e-6;
    for (size_t i = 0; i < vec.size(); ++i) {
        invWeights[i] = 1.0 / (vec[i] + epsilon);
        totalWeight += invWeights[i];
    }
    if (totalWeight <= 0.0) {
        // 全要素が非正の場合は常に0
        return 0;
    }

    // 0～totalWeight の一様乱数を生成
    static thread_local std::mt19937 engine{std::random_device{}()};
    std::uniform_real_distribution<double> dist(0.0, totalWeight);
    double r = dist(engine);

    // 累積和でインデックスを選択
    double cum = 0.0;
    for (size_t i = 0; i < invWeights.size(); ++i) {
        cum += invWeights[i];
        if (r <= cum) {
            return i;
        }
    }
    // 数値誤差対策
    return invWeights.size() - 1;
}

class PartitioningProblem : public SingleObjectiveProblem<SolutionType> {
   public:
    PartitioningProblem(int divisionsNumOfWeightVector, int objectivesNum,
                        int neighborhoodSize, int parallelSize)
        : _divisionsNumOfWeightVector(divisionsNumOfWeightVector),
          _objectivesNum(objectivesNum),
          _neighborhoodSize(neighborhoodSize),
          _parallelSize(parallelSize),
          _moeadInitializer() {}

    double ComputeObjective(const SolutionType& solution) override {
        auto graph = GenerateGraph(_divisionsNumOfWeightVector, _objectivesNum,
                                   _neighborhoodSize, _parallelSize,
                                   _moeadInitializer, solution);

        double averageDegree = 0.0;
        for (SimpleGraph::Node i = 0; i < graph.NodesNum(); ++i) {
            averageDegree += static_cast<double>(graph.Degree(i));
        }
        averageDegree /= static_cast<double>(graph.NodesNum());

        return averageDegree;
    }

   private:
    int _divisionsNumOfWeightVector;
    int _objectivesNum;
    int _neighborhoodSize;
    int _parallelSize;
    MoeadInitializer _moeadInitializer;
};

class NeighborGen : public NeighborGenerator<SolutionType> {
   public:
    NeighborGen(const std::vector<std::vector<double>>& avgDists)
        : avgDists(avgDists), _rng() {}

    SolutionType Generate(const SolutionType& currentSolution) override {
        constexpr int maxExchangeNodeTimes = 10;
        // constexpr int maxExchangeVectorTimes = 10;

        auto neighbor = currentSolution;
        // 重みベクトルを交換するノード数を決定
        // int times = _rng.Integer(1, maxExchangeNodeTimes);
        int times = 1;
        // 交換する片方のノードを選択
        auto firsts =
            _rng.Integers(0, currentSolution.size() - 1, times, false);
        for (auto first : firsts) {
            // 交換するもう片方のノードを選択
            auto second = SelectIndexByWeightedRandom(avgDists[first]);
            // 重みベクトルの交換回数の上限を決定
            auto max =
                std::min(static_cast<int>(currentSolution[first].size()),
                         static_cast<int>(currentSolution[second].size()));
            // 重みベクトルの交換回数を決定
            // times = _rng.Integer(1, max - 1);
            times = 1;
            // 交換する重みベクトルのインデックスを選択
            auto firstIndexes = _rng.Integers(
                0, currentSolution[first].size() - 1, times, false);
            auto secondIndexes = _rng.Integers(
                0, currentSolution[second].size() - 1, times, false);
            // 重みベクトルの交換
            for (int i = 0; i < times; ++i) {
                std::swap(neighbor[first][firstIndexes[i]],
                          neighbor[second][secondIndexes[i]]);
            }
        }
        return neighbor;
    }

   private:
    std::vector<std::vector<double>> avgDists;
    Rng _rng;
};

/// @brief グラフの隣接リストを表示する
void PrintAdjacencyList(const SimpleGraph& graph) {
    const auto& adjacencyList = graph.AdjacencyList();
    for (size_t i = 0; i < adjacencyList.size(); ++i) {
        std::cout << "Node " << i << ": ";
        for (const auto& neighbor : adjacencyList[i]) {
            std::cout << neighbor << " ";
        }
        std::cout << std::endl;
    }
}

/// @brief pythonの3次元リストのように表示する
void PrintSolution(const SolutionType& solution) {
    std::cout << "[\n";
    for (size_t i = 0; i < solution.size(); ++i) {
        std::cout << "  [";
        for (size_t j = 0; j < solution[i].size(); ++j) {
            std::cout << "[";
            for (size_t k = 0; k < solution[i][j].size(); ++k) {
                std::cout << solution[i][j][k];
                if (k != solution[i][j].size() - 1) std::cout << ", ";
            }
            std::cout << "]";
            if (j != solution[i].size() - 1) std::cout << ", ";
        }
        std::cout << "]";
        if (i != solution.size() - 1) std::cout << ",\n";
    }
    std::cout << "\n]" << std::endl;
}

int main() {
    int divisionsNumOfWeightVector = 8;
    int objectivesNum = 5;
    int neighborhoodSize = 21;
    int parallelSize = 50;
    MoeadInitializer moeadInitializer;

    // 初期解を生成
    auto vectorDivisions = moeadInitializer.GenerateWeightVectorDivisions(
        divisionsNumOfWeightVector, objectivesNum);
    auto initialSolution = moeadInitializer.LinearPartitioning(
        parallelSize, divisionsNumOfWeightVector, vectorDivisions);
    auto avgDists = ComputeDistanceMatrix(initialSolution);

    // SA構成クラスの生成
    auto problem = std::make_unique<PartitioningProblem>(
        divisionsNumOfWeightVector, objectivesNum, neighborhoodSize,
        parallelSize);
    auto neighborGen = std::make_unique<NeighborGen>(avgDists);

    // 初期化の評価
    auto initialGraph = GenerateGraph(divisionsNumOfWeightVector, objectivesNum,
                                      neighborhoodSize, parallelSize,
                                      moeadInitializer, initialSolution);
    auto initialObjective = problem->ComputeObjective(initialSolution);

    // Simulated Annealingの設定
    double initialTemperature = 0.1;
    double coolingRate = 0.99;
    double minTemperature = 1.e-6;
    int maxIterationsPerTemp = 10;
    long long maxTotalIterations = -1;
    int maxStagnantIterations = 5000000;
    bool verbose = true;
    SA<SolutionType> sa(initialSolution, std::move(problem),
                        std::move(neighborGen), initialTemperature, coolingRate,
                        minTemperature, maxIterationsPerTemp,
                        maxTotalIterations, maxStagnantIterations, verbose);
    // Simulated Annealingの実行
    auto result = sa.Run();

    // 最良解のグラフを生成
    auto graph = GenerateGraph(divisionsNumOfWeightVector, objectivesNum,
                               neighborhoodSize, parallelSize, moeadInitializer,
                               result.best);

    // 初期解と最良解の表示
    auto bestObjective = result.objective;
    std::cout << "Initial Objective: " << initialObjective
              << " max degree: " << initialGraph.MaxDegree() << std::endl;
    std::cout << "Best Objective: " << bestObjective
              << " max degree: " << graph.MaxDegree() << std::endl;

    // 最良解とそのグラフの隣接リストを表示
    std::cout << "Best Solution:" << std::endl;
    PrintSolution(result.best);
    std::cout << "Adjacency List:" << std::endl;
    PrintAdjacencyList(graph);
}