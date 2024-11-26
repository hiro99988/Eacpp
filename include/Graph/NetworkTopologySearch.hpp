#pragma once

#include <fstream>
#include <vector>

#include "Algorithms/MoeadInitializer.h"
#include "Graph/SimpleGraph.hpp"
#include "Rng/IRng.h"
#include "Rng/Rng.h"

namespace Eacpp {

class NetworkTopologySearch {
   public:
    NetworkTopologySearch(int objectivesNum, int neighborhoodSize, int divisionsNumOfWeightVector, int nodesNum,
                          int idealPathLengthBetweenNeighbors, int idealPathLengthBetweenExtremesAndAnyNode, int repeats,
                          double initialTemperature, double minTemperature, double coolingRate)
        : _objectivesNum(objectivesNum),
          _neighborhoodSize(neighborhoodSize),
          _divisionsNumOfWeightVector(divisionsNumOfWeightVector),
          _nodesNum(nodesNum),
          _idealPathLengthBetweenNeighbors(idealPathLengthBetweenNeighbors),
          _idealPathLengthBetweenExtremesAndAnyNode(idealPathLengthBetweenExtremesAndAnyNode),
          _repeats(repeats),
          _initialTemperature(initialTemperature),
          _minTemperature(minTemperature),
          _coolingRate(coolingRate),
          _moeadInitializer(MoeadInitializer()),
          _rng(std::make_unique<Rng>()) {}

    // NetworkTopologySearch(int objectivesNum, int neighborhoodSize, int divisionsNumOfWeightVector, int nodesNum,
    //                       int idealPathLengthBetweenNeighbors, int idealPathLengthBetweenExtremesAndAnyNode, int repeats,
    //                       double initialTemperature, double minTemperature, double coolingRate, SimpleGraph<int>
    //                       initialGraph)
    //     : _objectivesNum(objectivesNum),
    //       _neighborhoodSize(neighborhoodSize),
    //       _divisionsNumOfWeightVector(divisionsNumOfWeightVector),
    //       _nodesNum(nodesNum),
    //       _idealPathLengthBetweenNeighbors(idealPathLengthBetweenNeighbors),
    //       _idealPathLengthBetweenExtremesAndAnyNode(idealPathLengthBetweenExtremesAndAnyNode),
    //       _repeats(repeats),
    //       _initialTemperature(initialTemperature),
    //       _minTemperature(minTemperature),
    //       _coolingRate(coolingRate),
    //       _moeadInitializer(MoeadInitializer()),
    //       _rng(std::make_unique<Rng>()),
    //       _bestSoFarGraph(initialGraph) {
    //     _bestSoFarObjective = ComputeObjective(initialGraph);
    // }

    void Run();
    void Initialize();
    void Search();
    void WriteAdjacencyListToCsv(const std::ofstream&) const;

   private:
    int _objectivesNum;
    int _neighborhoodSize;
    int _divisionsNumOfWeightVector;
    int _nodesNum;
    int _idealPathLengthBetweenNeighbors;
    int _idealPathLengthBetweenExtremesAndAnyNode;
    int _repeats;
    double _initialTemperature;
    double _minTemperature;
    double _coolingRate;
    std::vector<std::vector<int>> _neighborhood;
    std::vector<int> _extremeNodes;
    MoeadInitializer _moeadInitializer;
    std::unique_ptr<IRng> _rng;
    double _bestSoFarObjective;
    SimpleGraph<int> _bestSoFarGraph;

    void InitializeNeighborhood();
    double ComputeObjective(const SimpleGraph<int>& graph) const;
    bool AcceptanceCriterion(double newObjective, double oldObjective, double temperature) const;
    void UpdateTemperature(double& temperature) const;
};

}  // namespace Eacpp