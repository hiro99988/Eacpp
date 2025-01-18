#include "Graph/NetworkTopologySearch.hpp"

#include <algorithm>
#include <array>
#include <eigen3/Eigen/Core>
#include <filesystem>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "Algorithms/MoeadInitializer.h"
#include "Graph/SimpleGraph.hpp"
#include "Rng/IRng.h"
#include "Utils/FileUtils.h"
#include "Utils/MpiUtils.h"
#include "Utils/Utils.h"

namespace Eacpp {

void NetworkTopologySearch::Run() {
    Initialize();
    Search();
    if (_isOutput) {
        Write();
    }
}

void NetworkTopologySearch::Initialize() {
    auto initialGraph = SimpleGraph::RandomRegularGraph(_nodesNum, _degree);
    _bestSoFarObjective = Evaluate(initialGraph);
    _bestSoFarGraph = std::move(initialGraph);
}

void NetworkTopologySearch::Search() {
    double bestObjective = _bestSoFarObjective;
    auto bestNeighbor = _bestSoFarGraph;
    double temperature = _initialTemperature;
    double lowerBoundOfAspl = LowerBoundOfAspl(_nodesNum, _degree);
    double unitOfAspl = UnitOfAspl(_nodesNum);

    for (int i = 0; i < _repeats; i++) {
        auto neighbor = bestNeighbor;

        while (true) {
            auto parents = _rng->Integers(0, neighbor.NodesNum() - 1, 2, false);
            std::array<std::set<Node>, 2> parentsNeighbors = {
                neighbor.Neighbors(parents[0]), neighbor.Neighbors(parents[1])};

            std::array<std::vector<Node>, 2> childrenCandidates;
            for (int j = 0; j < 2; j++) {
                for (auto&& k : parentsNeighbors[j]) {
                    if (k != parents[1 - j] &&
                        parentsNeighbors[1 - j].find(k) ==
                            parentsNeighbors[1 - j].end()) {
                        childrenCandidates[j].push_back(k);
                    }
                }
            }

            if (childrenCandidates[0].empty() ||
                childrenCandidates[1].empty()) {
                continue;
            }

            int child1Index = _rng->Integer(childrenCandidates[0].size() - 1);
            int child2Index = _rng->Integer(childrenCandidates[1].size() - 1);
            int child1 = childrenCandidates[0][child1Index];
            int child2 = childrenCandidates[1][child2Index];

            neighbor.TwoOpt(parents[0], child1, parents[1], child2);

            break;
        }

        double aspl = Evaluate(neighbor);
        // auto objective = aspl - lowerBoundOfAspl;
        auto objective = aspl;
        if (AcceptanceCriterion(objective, bestObjective, lowerBoundOfAspl,
                                unitOfAspl, temperature)) {
            bestObjective = objective;
            bestNeighbor = neighbor;
            if (objective < _bestSoFarObjective) {
                _bestSoFarObjective = objective;
                _bestSoFarGraph = bestNeighbor;
            }
            std::cout << i << " aspl: " << aspl << " diff from lower bound: "
                      << objective - lowerBoundOfAspl << std::endl;
        }

        UpdateTemperature(temperature, _minTemperature, _coolingRate);
    }

    std::cout << "Best So Far Objective: " << _bestSoFarObjective << std::endl;
    std::cout << "Best So Far Graph: " << std::endl;
    auto adjacencyList = _bestSoFarGraph.AdjacencyList();
    for (int i = 0; i < adjacencyList.size(); i++) {
        std::cout << "Node " << i << " neighbors: ";
        for (const auto& neighbor : adjacencyList[i]) {
            std::cout << neighbor << " ";
        }
        std::cout << std::endl;
    }
}

void NetworkTopologySearch::Write() const {
    // Create directory path
    std::ostringstream oss;
    oss << "n" << _nodesNum << "_d" << _degree;
    std::string directoryName = oss.str();
    std::filesystem::path directoryPath = "data/graphs/" + directoryName;

    // if the known graph is better than the discovered graph, skip writing
    // resultsS else create directory
    if (std::filesystem::exists(directoryPath)) {
        std::ifstream evalFile(directoryPath / "evaluation.json");
        if (evalFile.is_open()) {
            nlohmann::json json;
            evalFile >> json;
            double objective = json["objective"];

            if (_bestSoFarObjective >= objective) {
                std::cout << "Known graph is better than the discovered graph. "
                             "Skipping writing results."
                          << std::endl;
                return;
            }
        }
    } else {
        std::filesystem::create_directories(directoryPath);
    }

    // Write adjacencyList
    auto adjacencyList = _bestSoFarGraph.AdjacencyList();
    std::ofstream adjacencyListFile(directoryPath / "adjacencyList.csv");
    WriteCsv(adjacencyListFile, adjacencyList);

    // Write evaluation and parameters
    nlohmann::json evalJson;
    evalJson["objective"] = _bestSoFarObjective;
    evalJson["nodesNum"] = _nodesNum;
    evalJson["degree"] = _degree;
    evalJson["repeats"] = _repeats;
    evalJson["initialTemperature"] = _initialTemperature;
    evalJson["minTemperature"] = _minTemperature;
    evalJson["coolingRate"] = _coolingRate;
    double lowerBoundOfAspl = LowerBoundOfAspl(_nodesNum, _degree);
    evalJson["lowerBoundOfAspl"] = lowerBoundOfAspl;
    evalJson["diffenceFromLowerBound"] = _bestSoFarObjective - lowerBoundOfAspl;

    std::ofstream evalFile(directoryPath / "evaluation.json");
    evalFile << evalJson.dump(4);

    std::cout << "Results written to " << directoryPath << std::endl;
}

double NetworkTopologySearch::BestSoFarObjective() const {
    return _bestSoFarObjective;
}

SimpleGraph NetworkTopologySearch::BestSoFarGraph() const {
    return _bestSoFarGraph;
}

double NetworkTopologySearch::Evaluate(const SimpleGraph& graph) const {
    return graph.AverageShortestPathLength();
}

bool NetworkTopologySearch::AcceptanceCriterion(double newObjective,
                                                double oldObjective,
                                                double lowerBoundOfAspl,
                                                double unitOfAspl,
                                                double temperature) const {
    // double newGap = newObjective - lowerBoundOfAspl;
    // double oldGap = oldObjective - lowerBoundOfAspl;
    // double delta = newGap - oldGap;
    double delta = newObjective - oldObjective;
    return delta <= 0 || _rng->Random() < std::exp(-delta / (temperature));
}

void NetworkTopologySearch::UpdateTemperature(double& temperature,
                                              double minTemperature,
                                              double coolingRate) const {
    temperature *= coolingRate;
    temperature = std::max(temperature, minTemperature);
}

double NetworkTopologySearch::UnitOfAspl(int nodesNum) const {
    return 2.0 / (nodesNum * (nodesNum - 1));
}

double NetworkTopologySearch::LowerBoundOfAspl(int nodesNum, int degree) const {
    int diameter = -1;
    double aspl = 0.0;
    int n = 1;
    int i = 1;
    while (true) {
        double tmp = n + degree * std::pow(degree - 1, i - 1);
        if (tmp >= nodesNum) {
            break;
        }

        n = tmp;
        aspl += i * degree * std::pow(degree - 1, i - 1);
        diameter = i;
        i++;
    }

    diameter++;
    aspl += diameter * (nodesNum - n);
    aspl /= nodesNum - 1;

    return aspl;
}

}  // namespace Eacpp