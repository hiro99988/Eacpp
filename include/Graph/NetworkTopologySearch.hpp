#pragma once

#include <eigen3/Eigen/Core>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "Algorithms/MoeadInitializer.h"
#include "Graph/SimpleGraph.hpp"
#include "Rng/IRng.h"
#include "Rng/Rng.h"

namespace Eacpp {

class NetworkTopologySearch {
   private:
    using Node = SimpleGraph::Node;

   public:
    struct Evaluation {
        Evaluation() {}
        Evaluation(double objective, double asplNeighbors, double asplExtremes)
            : objective(objective),
              asplNeighbors(asplNeighbors),
              asplExtremes(asplExtremes) {}

        double objective = 0.0;
        double asplNeighbors = 0.0;
        double asplExtremes = 0.0;

        bool operator<(const Evaluation& other) const;
        bool operator>(const Evaluation& other) const;
        friend std::ostream& operator<<(std::ostream& os,
                                        const Evaluation& eval);
    };

   public:
    NetworkTopologySearch(int objectivesNum, int neighborhoodSize,
                          int divisionsNumOfWeightVector, int nodesNum,
                          int degree, double weightOfAsplNeighborsInObjective,
                          double weightOfAsplExtremesInObjective,
                          bool isOutput = true)
        : _objectivesNum(objectivesNum),
          _neighborhoodSize(neighborhoodSize),
          _divisionsNumOfWeightVector(divisionsNumOfWeightVector),
          _nodesNum(nodesNum),
          _degree(degree),
          _weightOfAsplNeighborsInObjective(weightOfAsplNeighborsInObjective),
          _weightOfAsplExtremesInObjective(weightOfAsplExtremesInObjective),
          _isOutput(isOutput),
          _rng(std::make_unique<Rng>()) {}

    void Run(int repeats, double initialTemperature, double minTemperature,
             double coolingRate);
    void Initialize();
    void Search(int repeats, double initialTemperature, double minTemperature,
                double coolingRate);
    void Analyze(std::map<Node, int>& outNeighborFrequency,
                 std::map<Node, int>& outExtremeFrequency) const;
    void Write(const std::map<Node, int>& neighborFrequency,
               const std::map<Node, int>& extremeFrequency) const;
    void WriteAdjacencyListToCsv(std::ofstream&) const;
    Evaluation BestSoFarEvaluation() const;
    SimpleGraph BestSoFarGraph() const;
    Evaluation Evaluate(const SimpleGraph& graph) const;
    void EvaluateSqls(const SimpleGraph& graph,
                      std::vector<std::vector<Node>>& outSplNeighbors,
                      std::vector<std::vector<Node>>& outSplExtremes) const;

   private:
    int _objectivesNum;
    int _neighborhoodSize;
    int _divisionsNumOfWeightVector;
    int _nodesNum;
    int _degree;
    double _weightOfAsplNeighborsInObjective;
    double _weightOfAsplExtremesInObjective;
    bool _isOutput;
    std::vector<Eigen::ArrayXd> _weightVectors;
    std::vector<std::vector<int>> _individualNeighborhoods;
    std::vector<std::vector<int>> _allNodeIndexes;
    std::vector<std::vector<int>> _nodeNeighbors;
    std::vector<int> _extremeNodes;
    MoeadInitializer _moeadInitializer;
    std::unique_ptr<IRng> _rng;
    Evaluation _bestSoFarEvaluation;
    SimpleGraph _bestSoFarGraph;

    void InitializeNodes();

    bool AcceptanceCriterion(double newObjective, double oldObjective,
                             double temperature) const;
    void UpdateTemperature(double& temperature, double minTemperature,
                           double coolingRate) const;
};

}  // namespace Eacpp