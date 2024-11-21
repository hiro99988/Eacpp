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
        : objectivesNum(objectivesNum),
          neighborhoodSize(neighborhoodSize),
          idealHopsNumBetweenNeighbors(idealHopsNumBetweenNeighbors),
          idealHopsNumBetweenExtremesAndAnyNode(idealHopsNumBetweenExtremesAndAnyNode),
          neighborhood(neighborhood),
          extremeNodes(extremeNodes),
          graph(graph) {}

    void Search() const;

   private:
    int objectivesNum;
    int neighborhoodSize;
    int idealHopsNumBetweenNeighbors;
    int idealHopsNumBetweenExtremesAndAnyNode;
    std::vector<std::vector<int>> neighborhood;
    std::vector<int> extremeNodes;
    SimpleGraph<int> graph;
    MoeadInitializer moeadInitializer;

    void InitializeNeighborhood() const;
};

}  // namespace Eacpp