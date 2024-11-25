#include "Graph/NetworkTopologySearch.hpp"

#include <algorithm>
#include <vector>

#include "Algorithms/MoeadInitializer.h"

namespace Eacpp {

void NetworkTopologySearch::Search() const {
    InitializeNeighborhood();

    std::vector<int> distances(_graph.NodesNum(), -1);
    std::vector<int> previous(_graph.NodesNum(), -1);

    std::queue<int> queue;
    queue.push(_extremeNodes[0]);
    distances[_extremeNodes[0]] = 0;

    while (!queue.empty()) {
        int node = queue.front();
        queue.pop();

        for (int i = 0; i < _neighborhoodSize; ++i) {
            int neighbor = _neighborhood[node][i];

            if (distances[neighbor] == -1) {
                distances[neighbor] = distances[node] + 1;
                previous[neighbor] = node;
                queue.push(neighbor);
            }
        }
    }

    int current = _extremeNodes[1];
    std::vector<int> path;

    while (current != -1) {
        path.push_back(current);
        current = previous[current];
    }

    std::reverse(path.begin(), path.end());

    for (int node : path) {
        std::cout << node << ' ';
    }

    std::cout << std::endl;
}

void NetworkTopologySearch::InitializeNeighborhood() const {
    std::vector<std::vector<int>> neighborhoods;
    neighborhoods = _moeadInitializer.CalculateNeighborhoods2d(
        _neighborhoodSize, _moeadInitializer.GenerateWeightVectors(_neighborhoodSize, _objectivesNum));

    for (int i = 0; i < neighborhoods.size(); ++i) {
        for (int j = 0; j < neighborhoods[i].size(); ++j) {
            if (neighborhoods[i][j] == i) {
                neighborhoods[i].erase(neighborhoods[i].begin() + j);
                break;
            }
        }
    }
}

}  // namespace Eacpp