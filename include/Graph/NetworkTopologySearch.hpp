#pragma once

#include <vector>

#include "Algorithms/MoeadInitializer.h"
#include "Graph/SimpleGraph.hpp"

namespace Eacpp {

class NetworkTopologySearch {
   public:
    NetworkTopologySearch(int objectivesNum, int neighborhoodSize, int idealHopsNumBetweenNeighbors,
                          int idealHopsNumBetweenExtremesAndAnyNode, const std::vector<std::vector<int>>& neighborhood,
                          const std::vector<int>& extremeNodes, const SimpleGraph<int>& graph)
        : _objectivesNum(objectivesNum),
          _neighborhoodSize(neighborhoodSize),
          _idealHopsNumBetweenNeighbors(idealHopsNumBetweenNeighbors),
          _idealHopsNumBetweenExtremesAndAnyNode(idealHopsNumBetweenExtremesAndAnyNode),
          _neighborhood(neighborhood),
          _extremeNodes(extremeNodes),
          _graph(graph) {}

    void Search() const;

   private:
    int _objectivesNum;
    int _neighborhoodSize;
    int _idealHopsNumBetweenNeighbors;
    int _idealHopsNumBetweenExtremesAndAnyNode;
    std::vector<std::vector<int>> _neighborhood;
    std::vector<int> _extremeNodes;
    SimpleGraph<int> _graph;
    MoeadInitializer _moeadInitializer;

    void InitializeNeighborhood() const;
};

}  // namespace Eacpp