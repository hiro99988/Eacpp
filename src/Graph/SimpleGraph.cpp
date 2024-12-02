#include "Graph/SimpleGraph.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Rng/Rng.h"

namespace Eacpp {

template <typename T>
size_t SimpleGraph<T>::ElementsNum(int nodesNum) {
    return nodesNum * (nodesNum - 1) / 2;
}

template <typename T>
SimpleGraph<T> SimpleGraph<T>::EmptyGraph(int nodesNum) {
    SimpleGraph<T> graph(nodesNum);
    return graph;
}

template <typename T>
SimpleGraph<T> SimpleGraph<T>::GnpRandomGraph(int nodesNum, double probability) {
    if (probability < 0 || probability > 1) {
        throw std::invalid_argument("Probability must be in the range [0, 1]");
    }

    if (probability <= 0) {
        SimpleGraph<T> emptyGraph(nodesNum);
        return emptyGraph;
    }

    if (probability >= 1) {
        std::vector<T> matrix(ElementsNum(nodesNum), 1);
        SimpleGraph<T> completeGraph(nodesNum, matrix);
        return completeGraph;
    }

    Rng rng;
    SimpleGraph<T> graph(nodesNum);

    for (int i = 0; i < nodesNum; ++i) {
        for (int j = i + 1; j < nodesNum; ++j) {
            if (rng.Random() < probability) {
                graph(i, j) = 1;
            }
        }
    }

    return graph;
}

template <typename T>
SimpleGraph<T> SimpleGraph<T>::RandomRegularGraph(int nodesNum, int degree) {
    if (nodesNum * degree % 2 != 0) {
        throw std::invalid_argument("The value of (nodesNum * degree) must be even");
    }

    if (degree < 0 || degree >= nodesNum) {
        throw std::invalid_argument("Degree must be in the range [0, nodesNum)");
    }

    if (degree == 0) {
        SimpleGraph<T> emptyGraph(nodesNum);
        return emptyGraph;
    }

    SimpleGraph<T> graph(nodesNum);

    struct TryCreation {
        std::set<std::pair<int, int>> operator()(int nodesNum, int degree) {
            std::set<std::pair<int, int>> edges;
            std::vector<int> stubs;
            for (int i = 0; i < nodesNum; ++i) {
                stubs.insert(stubs.end(), degree, i);
            }

            while (!stubs.empty()) {
                std::map<int, int> potentialEdges;
                std::shuffle(stubs.begin(), stubs.end(), std::mt19937(std::random_device()()));
                auto stubIter = stubs.begin();
                while (stubIter != stubs.end()) {
                    int s1 = *stubIter++;
                    if (stubIter == stubs.end()) {
                        potentialEdges[s1]++;
                        break;
                    }

                    int s2 = *stubIter++;
                    if (s1 > s2) {
                        std::swap(s1, s2);
                    }

                    if (s1 != s2 && edges.find({std::min(s1, s2), std::max(s1, s2)}) == edges.end()) {
                        edges.insert({s1, s2});
                    } else {
                        potentialEdges[s1]++;
                        potentialEdges[s2]++;
                    }
                }

                if (!Suitable(edges, potentialEdges)) {
                    return {};  // 失敗
                }

                stubs.clear();
                for (const auto& kv : potentialEdges) {
                    stubs.insert(stubs.end(), kv.second, kv.first);
                }
            }

            return edges;
        }

        bool Suitable(const std::set<std::pair<int, int>>& edges, const std::map<int, int>& potentialEdges) {
            if (potentialEdges.empty()) {
                return true;
            }

            std::vector<int> nodes;
            for (const auto& kv : potentialEdges) {
                nodes.push_back(kv.first);
            }

            for (size_t i = 0; i < nodes.size(); ++i) {
                for (size_t j = i + 1; j < nodes.size(); ++j) {
                    int s1 = nodes[i];
                    int s2 = nodes[j];
                    if (s1 == s2) {
                        continue;
                    }

                    if (edges.find({std::min(s1, s2), std::max(s1, s2)}) == edges.end()) {
                        return true;
                    }
                }
            }

            return false;
        }
    } tryCreation;

    // エッジセットの生成を試行
    std::set<std::pair<int, int>> edges = tryCreation();
    int max_attempts = 10;
    while (edges.empty() && max_attempts-- > 0) {
        edges = tryCreation();
    }
    if (edges.empty()) {
        throw std::runtime_error("Failed to generate a random regular graph");
    }

    // エッジをグラフに追加
    for (const auto& edge : edges) {
        graph(edge.first, edge.second) = 1;
        graph(edge.second, edge.first) = 1;
    }

    return graph;
}

template <typename T>
typename std::vector<T>::reference SimpleGraph<T>::operator[](size_t index) {
    if (index >= _matrix.size()) {
        throw std::out_of_range("Index out of range " + std::to_string(index) + " >= " + std::to_string(_matrix.size()));
    }

    return _matrix[index];
}

template <typename T>
typename std::vector<T>::const_reference SimpleGraph<T>::operator[](size_t index) const {
    if (index >= _matrix.size()) {
        throw std::out_of_range("Index out of range " + std::to_string(index) + " >= " + std::to_string(_matrix.size()));
    }

    return _matrix[index];
}

template <typename T>
typename std::vector<T>::reference SimpleGraph<T>::operator()(size_t row, size_t col) {
    ValidateIndexes(row, col);
    ValidateEdge(row, col);

    return _matrix[Index(row, col)];
}

template <typename T>
typename std::vector<T>::const_reference SimpleGraph<T>::operator()(size_t row, size_t col) const {
    ValidateIndexes(row, col);
    ValidateEdge(row, col);

    return _matrix[Index(row, col)];
}

template <typename T>
void SimpleGraph<T>::resize(int nodesNum) {
    _nodesNum = nodesNum;
    _matrix.resize(ElementsNum(nodesNum), 0);
}

template <typename T>
int SimpleGraph<T>::ShortestPathLength(int start, int end) const {
    if (start == end) {
        return 0;
    }

    std::vector<int> distances(_nodesNum, -1);
    distances[start] = 0;

    std::deque<int> queue;
    queue.push_back(start);

    while (!queue.empty()) {
        int node = queue.front();
        queue.pop_front();

        for (int i = 0; i < _nodesNum; ++i) {
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

    for (int i = 0; i < _nodesNum; ++i) {
        int degree = 0;
        for (int j = 0; j < _nodesNum; ++j) {
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
double SimpleGraph<T>::AverageShortestPathLength() const {
    double sum = 0;
    int count = 0;

    for (int i = 0; i < _nodesNum; ++i) {
        for (int j = i + 1; j < _nodesNum; ++j) {
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
    std::vector<std::vector<int>> adjacencyList(_nodesNum);

    for (int i = 0; i < _nodesNum; ++i) {
        for (int j = 0; j < _nodesNum; ++j) {
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

    _matrix[Index(parent1, child1)] = 0;
    _matrix[Index(parent2, child2)] = 0;
    _matrix[Index(parent1, child2)] = 1;
    _matrix[Index(parent2, child1)] = 1;
}

template <typename T>
std::vector<int> SimpleGraph<T>::GetEdges(size_t node) const {
    ValidateIndexes(node);

    std::vector<int> edges;

    for (int i = 0; i < _nodesNum; ++i) {
        if (node != i && Element(node, i)) {
            edges.push_back(i);
        }
    }

    return edges;
}

template <typename T>
size_t SimpleGraph<T>::Index(size_t row, size_t col) const {
    ValidateIndexes(row, col);
    ValidateEdge(row, col);

    if (row > col) {
        std::swap(row, col);
    }

    return row * (_nodesNum - 1) - (row * (row + 1) / 2) + col - 1;
}

template <typename T>
T SimpleGraph<T>::Element(size_t row, size_t col) const {
    ValidateIndexes(row, col);

    if (row == col) {
        return 0;
    }

    return _matrix[Index(row, col)];
}

template <typename T>
template <typename... Args>
void SimpleGraph<T>::ValidateIndexes(Args... indexes) const {
    for (size_t index : {indexes...}) {
        if (index >= _nodesNum) {
            throw std::out_of_range("Index out of range " + std::to_string(index) + " >= " + std::to_string(_nodesNum) +
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
