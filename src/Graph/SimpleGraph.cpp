#include "Graph/SimpleGraph.hpp"

#include <algorithm>
#include <deque>
#include <stdexcept>
#include <vector>

namespace Eacpp {

template <typename T>
typename std::vector<T>::reference SimpleGraph<T>::operator()(int row, int col) {
    ValidateIndexes(row, col);
    ValidateEdge(row, col);

    return matrix[Index(row, col)];
}

template <typename T>
typename std::vector<T>::const_reference SimpleGraph<T>::operator()(int row, int col) const {
    ValidateIndexes(row, col);
    ValidateEdge(row, col);

    return matrix[Index(row, col)];
}

template <typename T>
int SimpleGraph<T>::ShortestPathLength(int start, int end) const {
    if (start == end) {
        return 0;
    }

    std::vector<int> distances(nodesNum, -1);
    distances[start] = 0;

    std::deque<int> queue;
    queue.push_back(start);

    while (!queue.empty()) {
        int node = queue.front();
        queue.pop_front();

        for (int i = 0; i < nodesNum; ++i) {
            if (Element(node, i) && distances[i] == -1) {
                distances[i] = distances[node] + 1;
                queue.push_back(i);
            }
        }
    }

    return distances[end];
}

template <typename T>
int SimpleGraph<T>::MaxDegree() const {
    int maxDegree = 0;

    for (int i = 0; i < nodesNum; ++i) {
        int degree = 0;
        for (int j = 0; j < nodesNum; ++j) {
            if (Element(i, j)) {
                ++degree;
            }
        }

        if (degree > maxDegree) {
            maxDegree = degree;
        }
    }

    return maxDegree;
}

template <typename T>
int SimpleGraph<T>::AverageShortestPathLength() const {
    int sum = 0;
    int count = 0;

    for (int i = 0; i < nodesNum; ++i) {
        for (int j = i + 1; j < nodesNum; ++j) {
            int pathLength = ShortestPathLength(i, j);

            if (pathLength != -1) {
                sum += pathLength;
                ++count;
            }
        }
    }

    return sum / count;
}

template <typename T>
std::vector<std::vector<int>> SimpleGraph<T>::ToAdjacencyList() const {
    std::vector<std::vector<int>> adjacencyList(nodesNum);

    for (int i = 0; i < nodesNum; ++i) {
        for (int j = 0; j < nodesNum; ++j) {
            if (Element(i, j)) {
                adjacencyList[i].push_back(j);
            }
        }
    }

    return adjacencyList;
}

template <typename T>
void SimpleGraph<T>::TwoOpt(size_t parent1, size_t child1, size_t parent2, size_t child2) {
    ValidateIndexes(parent1, parent2, child1, child2);
    ValidateEdge(parent1, child1);
    ValidateEdge(parent2, child2);
    if (parent1 == parent2 || parent1 == child2 || parent2 == child1 || child1 == child2) {
        throw std::invalid_argument("No duplicate nodes allowed");
    }

    this[parent1, child1] = 0;
    this[parent2, child2] = 0;
    this[parent1, child2] = 1;
    this[parent2, child1] = 1;
}

template <typename T>
size_t SimpleGraph<T>::Index(size_t row, size_t col) const {
    ValidateIndexes(row, col);
    ValidateEdge(row, col);

    if (row > col) {
        std::swap(row, col);
    }

    return row * (nodesNum - 1) - (row * (row + 1) / 2) + col - 1;
}

template <typename T>
size_t SimpleGraph<T>::ElementsNum(int nodesNum) const {
    return nodesNum * (nodesNum - 1) / 2;
}

template <typename T>
T SimpleGraph<T>::Element(size_t row, size_t col) const {
    ValidateIndexes(row, col);

    if (row == col) {
        return 0;
    }

    return matrix[Index(row, col)];
}

template <typename T>
template <typename... Args>
    requires std::same_as<Args..., size_t>
void SimpleGraph<T>::ValidateIndexes(Args... indexes) const {
    for (size_t index : {indexes...}) {
        if (index >= nodesNum) {
            throw std::out_of_range("Index out of range " + std::to_string(index) + " >= " + std::to_string(nodesNum) +
                                    " nodesNum");
        }
    }
}

template <typename T>
void SimpleGraph<T>::ValidateEdge(size_t row, size_t col) const {
    if (row == col) {
        throw std::invalid_argument("No self-edges allowed " + std::to_string(row) + " == " + std::to_string(col));
    }
}

}  // namespace Eacpp
