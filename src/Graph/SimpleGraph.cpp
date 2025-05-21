#include "Graph/SimpleGraph.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Rng/Rng.h"

namespace Eacpp {

SimpleGraph SimpleGraph::FromEdges(Node nodesNum, const std::set<Edge>& edges) {
    SimpleGraph graph(nodesNum);

    for (const auto& edge : edges) {
        graph.AddEdge(edge);
    }

    return graph;
}

SimpleGraph SimpleGraph::EmptyGraph(Node nodesNum) {
    return SimpleGraph(nodesNum);
}

SimpleGraph SimpleGraph::GnpRandomGraph(Node nodesNum, double probability) {
    if (probability < 0 || probability > 1) {
        throw std::invalid_argument("Probability must be in the range [0, 1]");
    }

    if (probability <= 0) {
        return EmptyGraph(nodesNum);
    }

    Rng rng;
    SimpleGraph graph(nodesNum);

    for (int i = 0; i < nodesNum; ++i) {
        for (int j = i + 1; j < nodesNum; ++j) {
            if (rng.Random() < probability) {
                graph.AddEdge(i, j);
            }
        }
    }

    return graph;
}

SimpleGraph SimpleGraph::RandomRegularGraph(Node nodesNum, Node degree) {
    if (nodesNum * degree % 2 != 0) {
        throw std::invalid_argument(
            "The value of (nodesNum * degree) must be even");
    }

    if (degree < 0 || degree >= nodesNum) {
        throw std::invalid_argument(
            "Degree must be in the range [0, nodesNum)");
    }

    if (degree == 0) {
        return EmptyGraph(nodesNum);
    }

    struct TryCreation {
        /// @brief create edge set for a random degree-regular graph
        /// @return edge set. if the condition is not met, return an empty set.
        std::set<Edge> operator()(int nodesNum, int degree) {
            std::set<Edge> edges;
            // represent a half-edge to give each node a connection point
            std::vector<int> stubs;
            for (int i = 0; i < nodesNum; ++i) {
                stubs.insert(stubs.end(), degree, i);
            }

            // repeat until all stubs are connected
            while (!stubs.empty()) {
                // store the number of potential edges for each node
                std::map<int, int> potentialEdges;
                std::shuffle(stubs.begin(), stubs.end(),
                             std::mt19937(std::random_device()()));
                auto stubIter = stubs.begin();
                while (stubIter != stubs.end()) {
                    int s1 = *stubIter++;
                    int s2 = *stubIter++;
                    if (s1 > s2) {
                        std::swap(s1, s2);
                    }

                    if (s1 != s2 && edges.find({s1, s2}) == edges.end()) {
                        edges.insert({s1, s2});
                    } else {
                        // Need to record nodes that need to be reconnected to
                        // the edge because the edge could not be connected
                        potentialEdges[s1]++;
                        potentialEdges[s2]++;
                    }
                }

                if (!Suitable(edges, potentialEdges)) {
                    return {};  // failed to find suitable edge set
                }

                // update stubs with nodes that need to be reconnected
                stubs.clear();
                for (const auto& kv : potentialEdges) {
                    stubs.insert(stubs.end(), kv.second, kv.first);
                }
            }

            return edges;
        }

        /// @brief Check if there are suitable edges remaining.
        /// @return true if there are suitable edges remaining, false if the
        /// generation of the graph has failed.
        bool Suitable(const std::set<Edge>& edges,
                      const std::map<int, int>& potentialEdges) {
            if (potentialEdges.empty()) {
                return true;
            }

            std::vector<int> nodes;
            for (const auto& kv : potentialEdges) {
                nodes.push_back(kv.first);
            }

            // Check if there are any suitable edges left.
            for (auto&& s1 : nodes) {
                for (auto&& s2 : nodes) {
                    if (s1 == s2) {
                        break;
                    }

                    // if suitable edge found, return true
                    if (edges.find({std::min(s1, s2), std::max(s1, s2)}) ==
                        edges.end()) {
                        return true;
                    }
                }
            }

            return false;
        }
    } tryCreation;

    auto edges = tryCreation(nodesNum, degree);
    while (edges.empty()) {
        edges = tryCreation(nodesNum, degree);
    }

    return FromEdges(nodesNum, edges);
}

bool SimpleGraph::operator==(const SimpleGraph& other) const {
    return _nodesNum == other._nodesNum && _edges == other._edges &&
           _adjacencyList == other._adjacencyList;
}

bool SimpleGraph::operator!=(const SimpleGraph& other) const {
    return !(*this == other);
}

SimpleGraph::Node SimpleGraph::NodesNum() const {
    return _nodesNum;
}

const std::set<SimpleGraph::Edge>& SimpleGraph::Edges() const {
    return _edges;
}

const std::vector<std::set<SimpleGraph::Node>>& SimpleGraph::AdjacencyList()
    const {
    return _adjacencyList;
}

void SimpleGraph::AddEdge(Edge edge) {
    ValidateEdges(edge);
    NormalizeEdges(edge);

    if (_edges.insert(edge).second) {
        _adjacencyList[edge.first].insert(edge.second);
        _adjacencyList[edge.second].insert(edge.first);
    }
}

void SimpleGraph::AddEdge(Node u, Node v) {
    AddEdge({u, v});
}

void SimpleGraph::RemoveEdge(Edge edge) {
    ValidateEdges(edge);
    NormalizeEdges(edge);

    if (_edges.erase(edge) > 0) {
        _adjacencyList[edge.first].erase(edge.second);
        _adjacencyList[edge.second].erase(edge.first);
    }
}

void SimpleGraph::RemoveEdge(Node u, Node v) {
    RemoveEdge({u, v});
}

void SimpleGraph::Clear() {
    _edges.clear();
    for (auto& neighbors : _adjacencyList) {
        neighbors.clear();
    }
}

bool SimpleGraph::HasEdge(Edge edge) const {
    ValidateEdges(edge);
    NormalizeEdges(edge);

    return _edges.find(edge) != _edges.end();
}

bool SimpleGraph::HasEdge(Node u, Node v) const {
    return HasEdge({u, v});
}

std::set<SimpleGraph::Node> SimpleGraph::Neighbors(Node n) const {
    ValidateIndexes(n);

    return _adjacencyList[n];
}

std::vector<SimpleGraph::Node> SimpleGraph::Degrees() const {
    std::vector<Node> degrees(_nodesNum);

    for (Node i = 0; i < _nodesNum; ++i) {
        degrees[i] = _adjacencyList[i].size();
    }

    return degrees;
}

SimpleGraph::Node SimpleGraph::Degree(Node n) const {
    ValidateIndexes(n);

    return _adjacencyList[n].size();
}

SimpleGraph::Node SimpleGraph::MaxDegree() const {
    auto degrees = Degrees();
    return *std::max_element(degrees.begin(), degrees.end());
}

SimpleGraph::Node SimpleGraph::MinDegree() const {
    auto degrees = Degrees();
    return *std::min_element(degrees.begin(), degrees.end());
}

double SimpleGraph::AverageDegree() const {
    auto degrees = Degrees();
    double sum = std::reduce(degrees.begin(), degrees.end(), 0.0);
    return sum / _nodesNum;
}

void SimpleGraph::Resize(Node nodesNum) {
    _nodesNum = nodesNum;
    _edges.clear();
    _adjacencyList.resize(nodesNum);
}

SimpleGraph::Node SimpleGraph::ShortestPathLength(Node start, Node end) const {
    ValidateIndexes(start, end);

    if (start == end) {
        return 0;
    }

    std::deque<Node> queue;
    std::vector<int> distance(_nodesNum, -1);

    queue.push_back(start);
    distance[start] = 0;

    while (!queue.empty()) {
        Node current = queue.front();
        queue.pop_front();

        for (auto&& neighbor : _adjacencyList[current]) {
            if (distance[neighbor] == -1) {
                distance[neighbor] = distance[current] + 1;
                queue.push_back(neighbor);

                if (neighbor == end) {
                    return distance[neighbor];
                }
            }
        }
    }

    return -1;
}

double SimpleGraph::AverageShortestPathLength() const {
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

bool SimpleGraph::TwoOpt(Edge edge1, Edge edge2) {
    ValidateEdges(edge1, edge2);

    if (edge1.first == edge2.first || edge1.first == edge2.second ||
        edge1.second == edge2.first || edge1.second == edge2.second) {
        return false;
    }

    if (!HasEdge(edge1) || !HasEdge(edge2)) {
        return false;
    }

    RemoveEdge(edge1);
    RemoveEdge(edge2);
    AddEdge(edge1.first, edge2.second);
    AddEdge(edge2.first, edge1.second);

    return true;
}

bool SimpleGraph::TwoOpt(Node parent1, Node child1, Node parent2, Node child2) {
    return TwoOpt({parent1, child1}, {parent2, child2});
}

template <typename... Args>
void SimpleGraph::ValidateIndexes(Args... indexes) const {
    for (size_t index : {indexes...}) {
        if (index >= _nodesNum) {
            throw std::out_of_range(
                "Index out of range " + std::to_string(index) +
                " >= " + std::to_string(_nodesNum) + " nodesNum");
        }
    }
}

template <typename... Args>
void SimpleGraph::ValidateEdges(const Args&... edges) const {
    for (const auto& e : {edges...}) {
        ValidateIndexes(e.first, e.second);
        if (e.first == e.second) {
            throw std::invalid_argument("No self-edges allowed " +
                                        std::to_string(e.first) +
                                        " == " + std::to_string(e.second));
        }
    }
}

void SimpleGraph::NormalizeEdges(Edge& edge) const {
    if (edge.first > edge.second) {
        std::swap(edge.first, edge.second);
    }
}

}  // namespace Eacpp
