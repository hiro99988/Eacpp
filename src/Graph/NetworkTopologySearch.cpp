#include "Graph/NetworkTopologySearch.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "Algorithms/MoeadInitializer.h"
#include "Graph/SimpleGraph.hpp"
#include "Utils/MpiUtils.h"

namespace Eacpp {

void NetworkTopologySearch::Run() {
    Initialize();
    Search();
}

void NetworkTopologySearch::Initialize() {
    InitializeNeighborhood();
    auto initialGraph = SimpleGraph<int>::GnpRandomGraph(_nodesNum, 0.2);
    _bestSoFarObjective = ComputeObjective(initialGraph);
    _bestSoFarGraph = initialGraph;
}

void NetworkTopologySearch::Search() {
    double bestObjective = _bestSoFarObjective;
    auto bestNeighbor = _bestSoFarGraph;
    double temperature = _initialTemperature;
    for (int i = 0; i < _repeats; i++) {
        auto neighbor = bestNeighbor;

        double r = _rng->Random();
        if (r < 0.5) {
            auto index = _rng->Integer(neighbor.size() - 1);
            neighbor[index] = !neighbor[index];
        } else {
            auto indexes = _rng->Integers(0, neighbor.NodesNum() - 1, 2, false);
            auto child1 = _rng->Choice(neighbor.GetEdges(indexes[0]), 1, false);
            auto child2 = _rng->Choice(neighbor.GetEdges(indexes[1]), 1, false);
            while (child1[0] == indexes[1] || child2[0] == indexes[0] || child1[0] == child2[0]) {
                child1 = _rng->Choice(neighbor.GetEdges(indexes[0]), 1, false);
                child2 = _rng->Choice(neighbor.GetEdges(indexes[1]), 1, false);
            }

            neighbor.TwoOpt(indexes[0], indexes[1], child1[0], child2[0]);
        }

        double objective = ComputeObjective(neighbor);
        if (AcceptanceCriterion(objective, bestObjective, temperature)) {
            bestObjective = objective;
            bestNeighbor = neighbor;
            if (bestObjective < _bestSoFarObjective) {
                _bestSoFarObjective = bestObjective;
                _bestSoFarGraph = bestNeighbor;
            }

            // 表示
            std::cout << "i: " << i << " Objective: " << bestObjective << " maxDegree: " << bestNeighbor.MaxDegree()
                      << " temperature: " << temperature << std::endl;
        }

        UpdateTemperature(temperature);
    }
}

void NetworkTopologySearch::InitializeNeighborhood() {
    // int populationSize = _moeadInitializer.CalculatePopulationSize(_divisionsNumOfWeightVector, _objectivesNum);
    // auto weightVectors = _moeadInitializer.GenerateWeightVectors(_objectivesNum, _neighborhoodSize);
    // auto neighborhood = _moeadInitializer.CalculateNeighborhoods2d(_neighborhoodSize, weightVectors);
    // auto workloads = CalculateNodeWorkloads(populationSize, _nodesNum);

    // auto allNodeIndexes = GenerateAllNodeIndexes(populationSize, _nodesNum);

    // // 近傍を計算
    // _neighborhood.reverse(_nodesNum);

    _neighborhood = std::vector<std::vector<int>>(_nodesNum);
    _neighborhood[0] = {1, 2, 3};
    _neighborhood[1] = {0, 2, 3};
    _neighborhood[_nodesNum - 2] = {46, 47, 49};
    _neighborhood[_nodesNum - 1] = {46, 47, 48};
    for (int i = 2; i < _nodesNum - 2; i++) {
        _neighborhood[i] = {i - 2, i - 1, i + 1, i + 2};
    }

    _extremeNodes = {0, 49};
}

double NetworkTopologySearch::ComputeObjective(const SimpleGraph<int>& graph) const {
    // 近傍のノード間の最短経路長が理想値より大きい場合と極点ノードと任意のノード間の最短経路長が理想値より大きい場合の違反数を計算
    int violationsNumNeighborhood = 0;
    int violationsNumExtremes = 0;

    for (int i = 0; i < graph.NodesNum(); i++) {
        for (auto&& j : _neighborhood[i]) {
            if (i == j) {
                continue;
            }

            double shortestPathLength = graph.ShortestPathLength(i, j);
            if (shortestPathLength > _idealPathLengthBetweenNeighbors) {
                ++violationsNumNeighborhood;
            }
        }

        for (int j = 0; j < _extremeNodes.size(); j++) {
            double shortestPathLength = graph.ShortestPathLength(i, _extremeNodes[j]);
            if (shortestPathLength > _idealPathLengthBetweenExtremesAndAnyNode) {
                ++violationsNumExtremes;
            }
        }
    }

    return graph.MaxDegree() + violationsNumNeighborhood + violationsNumExtremes;
}

bool NetworkTopologySearch::AcceptanceCriterion(double newObjective, double oldObjective, double temperature) const {
    double delta = newObjective - oldObjective;
    return delta < 0 || _rng->Random() < std::exp(-delta / temperature);
}

void NetworkTopologySearch::UpdateTemperature(double& temperature) const {
    temperature *= _coolingRate;
    temperature = std::max(temperature, _minTemperature);
}

}  // namespace Eacpp