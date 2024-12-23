#pragma once

#include <cstdint>
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
    NetworkTopologySearch(int nodesNum, int degree, int repeats,
                          double initialTemperature, double minTemperature,
                          double coolingRate, bool isOutput = true)
        : _nodesNum(nodesNum),
          _degree(degree),
          _repeats(repeats),
          _initialTemperature(initialTemperature),
          _minTemperature(minTemperature),
          _coolingRate(coolingRate),
          _isOutput(isOutput),
          _rng(std::make_unique<Rng>()) {}

    NetworkTopologySearch(int nodesNum, int degree, int repeats,
                          double initialTemperature, double minTemperature,
                          double coolingRate, std::uint_fast32_t seed,
                          bool isOutput = true)
        : _nodesNum(nodesNum),
          _degree(degree),
          _repeats(repeats),
          _initialTemperature(initialTemperature),
          _minTemperature(minTemperature),
          _coolingRate(coolingRate),
          _isOutput(isOutput),
          _rng(std::make_unique<Rng>(seed)) {}

    void Run();
    void Initialize();
    void Search();
    void Write() const;
    double BestSoFarObjective() const;
    SimpleGraph BestSoFarGraph() const;
    double Evaluate(const SimpleGraph& graph) const;

   private:
    int _nodesNum;
    int _degree;
    int _repeats;
    double _initialTemperature;
    double _minTemperature;
    double _coolingRate;
    bool _isOutput;
    std::unique_ptr<IRng> _rng;
    double _bestSoFarObjective;
    SimpleGraph _bestSoFarGraph;

    bool AcceptanceCriterion(double newObjective, double oldObjective,
                             double temperature) const;
    void UpdateTemperature(double& temperature, double minTemperature,
                           double coolingRate) const;
};

}  // namespace Eacpp