#include "Problems/WeightVectorPartitioning.hpp"

#include <algorithm>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "Algorithms/MoeadInitializer.h"
#include "Algorithms/SimulatedAnnealing.hpp"
#include "Graph/SimpleGraph.hpp"
#include "Rng/Rng.h"
#include "Utils/Utils.h"

namespace Eacpp {

double PartitioningProblem::ComputeObjective(const SolutionType& solution) {
    auto graph = GenerateGraph(solution);

    return graph.AverageDegree();
}

SimpleGraph PartitioningProblem::GenerateGraph(const SolutionType& solution) {
    std::vector<int> ranksToSendAtInitialization;
    std::vector<int> ranksToSendCounts;
    std::vector<int> neighboringRanks;
    std::vector<int> neighboringRankCounts;
    GenerateAdjacencyList(solution, ranksToSendAtInitialization,
                          ranksToSendCounts, neighboringRanks,
                          neighboringRankCounts);

    SimpleGraph graph = SimpleGraph::EmptyGraph(ranksToSendCounts.size());
    // 各ノードがメッセージを送信するランクに基づいてエッジを追加する
    int sendIndex = 0;
    for (int i = 0; i < ranksToSendCounts.size(); ++i) {
        for (int j = 0; j < ranksToSendCounts[i]; ++j) {
            int target = ranksToSendAtInitialization[sendIndex++];
            graph.AddEdge(i, target);
        }
    }

    // 隣接受信情報に基づいてエッジを追加する
    int neighborIndex = 0;
    for (int i = 0; i < neighboringRankCounts.size(); ++i) {
        for (int j = 0; j < neighboringRankCounts[i]; ++j) {
            int neighbor = neighboringRanks[neighborIndex++];
            graph.AddEdge(neighbor, i);
        }
    }

    return graph;
}

void PartitioningProblem::GenerateAdjacencyList(
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
    _moeadInitializer.InitializeParallelMoead(
        _divisionsNumOfWeightVector, _objectivesNum, _neighborhoodSize,
        _parallelSize, internalIndividualIndexes, internalWeightVectors,
        internalNeighborhoods, internalIndividualCounts,
        externalIndividualIndexes, externalIndividualRanks,
        externalWeightVectors, externalIndividualCounts,
        outRanksToSendAtInitialization, outRanksToSendCounts,
        outNeighboringRanks, outNeighboringRankCounts, solution);
}

SolutionType NeighborGen::Generate(const SolutionType& currentSolution) {
    auto neighbor = currentSolution;

    // 交換する一つ目のノードを選択
    auto first = _rng.Integer(0, currentSolution.size() - 1);
    auto avgDist = ComputeDistanceMatrix(currentSolution, first);
    // 交換する二つ目のノードを選択
    auto second = SelectIndexByWeightedRandom(avgDist);
    // 交換する重みベクトルのインデックスを選択
    auto firstIndex = _rng.Integer(0, currentSolution[first].size() - 1);
    auto secondIndex = _rng.Integer(0, currentSolution[second].size() - 1);
    // 重みベクトルの交換
    std::swap(neighbor[first][firstIndex], neighbor[second][secondIndex]);

    return neighbor;
}

std::vector<double> NeighborGen::ComputeDistanceMatrix(
    const SolutionType& solution, std::size_t node) {
    auto n = solution.size();
    std::vector<double> avgDist(n, 0.0);

    for (size_t j = 0; j < n; ++j) {
        if (j == node) {
            continue;
        }

        double sum = 0.0;
        size_t count = 0;
        for (const auto& vecA : solution[node]) {
            for (const auto& vecB : solution[j]) {
                sum += CalculateSquaredEuclideanDistance(vecA, vecB);
                ++count;
            }
        }
        double avg = (count > 0) ? (sum / static_cast<double>(count)) : 0.0;
        avgDist[j] = avg;
    }

    return avgDist;
}

std::size_t NeighborGen::SelectIndexByWeightedRandom(
    const std::vector<double>& avgDist) {
    // 小さい値ほど選ばれやすくするため、重みを逆数で計算
    std::vector<double> invWeights(avgDist.size());
    double totalWeight = 0.0;
    for (size_t i = 0; i < avgDist.size(); ++i) {
        // 0の場合は選ばれないように0で初期化
        invWeights[i] = avgDist[i] > 0.0 ? 1.0 / avgDist[i] : 0.0;
        totalWeight += invWeights[i];
    }
    if (totalWeight <= 0.0) {
        // 全要素が非正の場合は常に0
        return 0;
    }

    // 0～totalWeight の一様乱数を生成
    double r = _rng.Uniform(0.0, totalWeight);

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

}  // namespace Eacpp