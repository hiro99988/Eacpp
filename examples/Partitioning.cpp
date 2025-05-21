#include <iostream>
#include <numeric>
#include <random>

#include "Algorithms/MoeadInitializer.h"
#include "Algorithms/SimulatedAnnealing.hpp"
#include "Graph/SimpleGraph.hpp"
#include "Problems/WeightVectorPartitioning.hpp"
#include "Rng/Rng.h"
#include "Utils/Utils.h"

using namespace Eacpp;

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
    // 問題設定
    int divisionsNumOfWeightVector = 23;
    int objectivesNum = 3;
    int neighborhoodSize = 21;
    int parallelSize = 50;

    // Simulated Annealingの設定
    double initialTemperature = 0.1;
    double coolingRate = 0.99;
    double minTemperature = 1.e-6;
    int maxIterationsPerTemp = 10;
    long long maxTotalIterations = 5'000'000;
    int maxStagnantIterations = -1;
    bool verbose = true;

    // 初期解を生成
    MoeadInitializer moeadInitializer;
    auto vectorDivisions = moeadInitializer.GenerateWeightVectorDivisions(
        divisionsNumOfWeightVector, objectivesNum);
    auto initialSolution = moeadInitializer.LinearPartitioning(
        parallelSize, divisionsNumOfWeightVector, vectorDivisions);

    // SA構成クラスの生成
    auto problem = std::make_unique<PartitioningProblem>(
        divisionsNumOfWeightVector, objectivesNum, neighborhoodSize,
        parallelSize);
    auto neighborGen = std::make_unique<NeighborGen>();

    // 初期化の評価
    auto initialGraph = problem->GenerateGraph(initialSolution);
    auto initialObjective = problem->ComputeObjective(initialSolution);

    // Simulated Annealingのインスタンスを生成
    SA<SolutionType> sa(initialSolution, std::move(problem),
                        std::move(neighborGen), initialTemperature, coolingRate,
                        minTemperature, maxIterationsPerTemp,
                        maxTotalIterations, maxStagnantIterations, verbose);
    // Simulated Annealingの実行
    auto result = sa.Run();

    // 最良解のグラフを生成
    problem = std::make_unique<PartitioningProblem>(
        divisionsNumOfWeightVector, objectivesNum, neighborhoodSize,
        parallelSize);
    auto graph = problem->GenerateGraph(result.best);

    // 初期解と最良解の表示
    auto bestObjective = result.objective;
    std::cout << "Initial Objective: " << initialObjective
              << " max degree: " << initialGraph.MaxDegree()
              << " min degree: " << initialGraph.MinDegree() << std::endl;
    std::cout << "Best Objective: " << bestObjective
              << " max degree: " << graph.MaxDegree()
              << " min degree: " << graph.MinDegree() << std::endl;

    // 最良解とそのグラフの隣接リストを表示
    std::cout << "Best Solution:" << std::endl;
    PrintSolution(result.best);
    std::cout << "Adjacency List:" << std::endl;
    PrintAdjacencyList(graph);
}