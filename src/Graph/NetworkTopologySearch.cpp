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

bool NetworkTopologySearch::Evaluation::operator<(
    const Evaluation& other) const {
    return objective < other.objective;
}

bool NetworkTopologySearch::Evaluation::operator>(
    const Evaluation& other) const {
    return other < *this;
}

std::ostream& operator<<(std::ostream& os,
                         const NetworkTopologySearch::Evaluation& eval) {
    os << "Objective: " << eval.objective
       << ", ASPL Neighbors: " << eval.asplNeighbors
       << ", ASPL Extremes: " << eval.asplExtremes;

    return os;
}

void NetworkTopologySearch::Run(int repeats, double initialTemperature,
                                double minTemperature, double coolingRate) {
    Initialize();
    Search(repeats, initialTemperature, minTemperature, coolingRate);
    std::map<Node, int> neighborFrequency;
    std::map<Node, int> extremeFrequency;
    Analyze(neighborFrequency, extremeFrequency);
    if (_isOutput) {
        Write(neighborFrequency, extremeFrequency);
    }
}

void NetworkTopologySearch::Initialize() {
    InitializeNodes();
    auto initialGraph = SimpleGraph::RandomRegularGraph(_nodesNum, _degree);
    _bestSoFarEvaluation = Evaluate(initialGraph);
    _bestSoFarGraph = std::move(initialGraph);
}

void NetworkTopologySearch::Search(int repeats, double initialTemperature,
                                   double minTemperature, double coolingRate) {
    double bestObjective = _bestSoFarEvaluation.objective;
    auto bestNeighbor = _bestSoFarGraph;
    double temperature = initialTemperature;

    for (int i = 0; i < repeats; i++) {
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

        auto evaluation = Evaluate(neighbor);
        if (AcceptanceCriterion(evaluation.objective, bestObjective,
                                temperature)) {
            bestObjective = evaluation.objective;
            bestNeighbor = neighbor;
            if (evaluation < _bestSoFarEvaluation) {
                _bestSoFarEvaluation = evaluation;
                _bestSoFarGraph = bestNeighbor;
            }
            std::cout << i << " " << evaluation << std::endl;
        }

        UpdateTemperature(temperature, minTemperature, coolingRate);
    }

    std::cout << "Best So Far Evaluation: " << _bestSoFarEvaluation
              << std::endl;
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

void NetworkTopologySearch::Analyze(
    std::map<Node, int>& outNeighborFrequency,
    std::map<Node, int>& outExtremeFrequency) const {
    for (Node node = 0; node < _bestSoFarGraph.NodesNum(); node++) {
        for (const auto& neighbor : _nodeNeighbors[node]) {
            auto distance = _bestSoFarGraph.ShortestPathLength(node, neighbor);
            outNeighborFrequency[distance]++;
        }
    }

    for (Node node = 0; node < _bestSoFarGraph.NodesNum(); node++) {
        for (const auto& extreme : _extremeNodes) {
            if (node != extreme) {
                double distance =
                    _bestSoFarGraph.ShortestPathLength(node, extreme);
                outExtremeFrequency[distance]++;
            }
        }
    }

    std::cout << "Neighbor Frequency: " << std::endl;
    for (const auto& [distance, frequency] : outNeighborFrequency) {
        std::cout << distance << ": " << frequency << std::endl;
    }

    std::cout << "Extreme Frequency: " << std::endl;
    for (const auto& [distance, frequency] : outExtremeFrequency) {
        std::cout << distance << ": " << frequency << std::endl;
    }
}

void NetworkTopologySearch::Write(
    const std::map<Node, int>& neighborFrequency,
    const std::map<Node, int>& extremeFrequency) const {
    // Create directory path
    std::ostringstream oss;
    std::string weightOfAsplNeighborsInObjectiveStr =
        ConvertDoubleToStringByDividingIntoIntegersAndDecimals(
            _weightOfAsplNeighborsInObjective);
    std::string weightOfAsplExtremesInObjectiveStr =
        ConvertDoubleToStringByDividingIntoIntegersAndDecimals(
            _weightOfAsplExtremesInObjective);
    oss << _objectivesNum << "_" << _neighborhoodSize << "_"
        << _divisionsNumOfWeightVector << "_" << _nodesNum << "_" << _degree
        << "_" << weightOfAsplNeighborsInObjectiveStr << "_"
        << weightOfAsplExtremesInObjectiveStr;
    std::string parameterPath = oss.str();
    std::filesystem::path directoryPath = "data/graph/" + parameterPath;

    // if the known graph is better than the discovered graph, skip writing
    // resultsS else create directory
    if (std::filesystem::exists(directoryPath)) {
        std::ifstream evalFile(directoryPath / "evaluation.json");
        if (evalFile.is_open()) {
            nlohmann::json json;
            evalFile >> json;
            double objective = json["objective"];

            if (!(_bestSoFarEvaluation.objective < objective)) {
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
    std::vector<std::vector<int>> binaryAdjacencyList;
    binaryAdjacencyList.reserve(adjacencyList.size());

    // Convert adjacencyList to binaryAdjacencyList
    // 1. If the neighbor is in _nodeNeighbors, push 1 and the neighbor
    // 2. If the neighbor is not in _nodeNeighbors, push 0 and the neighbor
    for (std::size_t i = 0; i < adjacencyList.size(); i++) {
        std::vector<int> binaryNeighbors;
        binaryNeighbors.reserve(adjacencyList[i].size() * 2);
        for (auto&& neighbor : adjacencyList[i]) {
            if (std::find(_nodeNeighbors[i].begin(), _nodeNeighbors[i].end(),
                          neighbor) != _nodeNeighbors[i].end()) {
                binaryNeighbors.push_back(1);
            } else {
                binaryNeighbors.push_back(0);
            }

            binaryNeighbors.push_back(neighbor);
        }

        binaryAdjacencyList.push_back(std::move(binaryNeighbors));
    }

    std::ofstream adjacencyListFile(directoryPath / "adjacencyList.csv");
    WriteCsv(adjacencyListFile, binaryAdjacencyList);

    // Write evaluation
    nlohmann::json evalJson;
    evalJson["objective"] = _bestSoFarEvaluation.objective;
    evalJson["asplNeighbors"] = _bestSoFarEvaluation.asplNeighbors;
    evalJson["asplExtremes"] = _bestSoFarEvaluation.asplExtremes;

    std::ofstream evalFile(directoryPath / "evaluation.json");
    evalFile << evalJson.dump(4);

    // Write SqlNeighbors and splExtremes
    std::vector<std::vector<Node>> splNeighbors;
    std::vector<std::vector<Node>> splExtremes;
    EvaluateSqls(_bestSoFarGraph, splNeighbors, splExtremes);

    constexpr std::array<const char*, 3> SplFileHeader = {"node", "neighbor",
                                                          "spl"};
    std::ofstream splNeighborsFile(directoryPath / "splNeighbors.csv");
    WriteCsv(splNeighborsFile, splNeighbors, SplFileHeader);

    constexpr std::array<const char*, 3> SplExtremesFileHeader = {
        "node", "extreme", "spl"};
    std::ofstream splExtremesFile(directoryPath / "splExtremes.csv");
    WriteCsv(splExtremesFile, splExtremes, SplExtremesFileHeader);

    // Write frequency
    constexpr std::array<const char*, 2> FrequencyFileHeader = {"distance",
                                                                "frequency"};
    std::ofstream neighborFrequencyFile(directoryPath /
                                        "neighborFrequency.csv");
    WriteCsvLine(neighborFrequencyFile, FrequencyFileHeader);
    for (const auto& [distance, frequency] : neighborFrequency) {
        neighborFrequencyFile << distance << "," << frequency << std::endl;
    }

    std::ofstream extremeFrequencyFile(directoryPath / "extremeFrequency.csv");
    WriteCsvLine(extremeFrequencyFile, FrequencyFileHeader);
    for (const auto& [distance, frequency] : extremeFrequency) {
        extremeFrequencyFile << distance << "," << frequency << std::endl;
    }

    std::cout << "Results written to " << directoryPath << std::endl;
}

NetworkTopologySearch::Evaluation NetworkTopologySearch::BestSoFarEvaluation()
    const {
    return _bestSoFarEvaluation;
}

SimpleGraph NetworkTopologySearch::BestSoFarGraph() const {
    return _bestSoFarGraph;
}

NetworkTopologySearch::Evaluation NetworkTopologySearch::Evaluate(
    const SimpleGraph& graph) const {
    double sumSplNeighbors = 0.0;
    int countSplNeighbors = 0;
    double sumSplExtremes = 0.0;
    int countSplExtremes = 0;

    for (Node i = 0; i < graph.NodesNum(); i++) {
        for (auto&& j : _nodeNeighbors[i]) {
            auto spl = graph.ShortestPathLength(i, j);
            sumSplNeighbors += spl;
            ++countSplNeighbors;
        }

        for (auto&& j : _extremeNodes) {
            if (i == j) {
                continue;
            }

            auto spl = graph.ShortestPathLength(i, j);
            sumSplExtremes += spl;
            ++countSplExtremes;
        }
    }

    double asplNeighbors = sumSplNeighbors / countSplNeighbors;
    double asplExtremes = sumSplExtremes / countSplExtremes;
    double objective = asplNeighbors * _weightOfAsplNeighborsInObjective +
                       asplExtremes * _weightOfAsplExtremesInObjective;

    return Evaluation(objective, asplNeighbors, asplExtremes);
}

void NetworkTopologySearch::EvaluateSqls(
    const SimpleGraph& graph, std::vector<std::vector<Node>>& outSplNeighbors,
    std::vector<std::vector<Node>>& outSplExtremes) const {
    outSplNeighbors.reserve(graph.NodesNum());
    outSplExtremes.reserve(graph.NodesNum());

    for (Node i = 0; i < graph.NodesNum(); i++) {
        for (auto&& j : _nodeNeighbors[i]) {
            auto spl = graph.ShortestPathLength(i, j);
            outSplNeighbors.push_back({i, static_cast<Node>(j), spl});
        }

        for (auto&& j : _extremeNodes) {
            if (i == j) {
                continue;
            }

            auto spl = graph.ShortestPathLength(i, j);
            outSplExtremes.push_back({i, static_cast<Node>(j), spl});
        }
    }
}

void NetworkTopologySearch::InitializeNodes() {
    int populationSize = _moeadInitializer.CalculatePopulationSize(
        _divisionsNumOfWeightVector, _objectivesNum);
    _moeadInitializer.GenerateWeightVectorsAndNeighborhoods(
        _divisionsNumOfWeightVector, _objectivesNum, _neighborhoodSize,
        _weightVectors, _individualNeighborhoods);
    _allNodeIndexes = GenerateAllNodeIndexes(populationSize, _nodesNum);

    std::vector<std::set<int>> allNeighborsNodeHas;
    allNeighborsNodeHas.reserve(_nodesNum);
    for (int i = 0; i < _nodesNum; i++) {
        std::set<int> neighbors;
        for (auto&& j : _allNodeIndexes[i]) {
            neighbors.insert(_individualNeighborhoods[j].begin(),
                             _individualNeighborhoods[j].end());
        }

        allNeighborsNodeHas.push_back(neighbors);
    }

    _nodeNeighbors.reserve(_nodesNum);
    for (int i = 0; i < _nodesNum; i++) {
        std::set<int> neighbors;
        for (auto&& j : allNeighborsNodeHas[i]) {
            int rank = GetRankFromIndex(populationSize, j, _nodesNum);
            if (rank != i) {
                neighbors.insert(rank);
            }
        }

        _nodeNeighbors.push_back(
            std::vector<int>(neighbors.begin(), neighbors.end()));
    }

    _extremeNodes.reserve(_objectivesNum);
    for (int i = 0; i < _nodesNum; i++) {
        for (auto&& j : _allNodeIndexes[i]) {
            int countOnes = (_weightVectors[j] == 1.0).count();
            int countZeros = (_weightVectors[j] == 0.0).count();
            if (countOnes == 1 && countZeros == _weightVectors[j].size() - 1) {
                _extremeNodes.push_back(i);
                break;
            }
        }
    }

    // // Display _nodeNeighbors
    // for (int i = 0; i < _nodeNeighbors.size(); ++i) {
    //     std::cout << "Node " << i << " neighbors: ";
    //     for (const auto& neighbor : _nodeNeighbors[i]) {
    //         std::cout << neighbor << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // // Display _extremeNodes
    // std::cout << "Extreme Nodes: ";
    // for (const auto& node : _extremeNodes) {
    //     std::cout << node << " ";
    // }
    // std::cout << std::endl;
}

bool NetworkTopologySearch::AcceptanceCriterion(double newObjective,
                                                double oldObjective,
                                                double temperature) const {
    double delta = newObjective - oldObjective;
    return delta < 0 || _rng->Random() < std::exp(-delta / temperature);
}

void NetworkTopologySearch::UpdateTemperature(double& temperature,
                                              double minTemperature,
                                              double coolingRate) const {
    temperature *= coolingRate;
    temperature = std::max(temperature, minTemperature);
}

}  // namespace Eacpp