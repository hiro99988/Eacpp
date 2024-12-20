#pragma once

#include <set>
#include <vector>

namespace Eacpp {

class SimpleGraph {
   public:
    using Node = std::size_t;
    using Edge = std::pair<Node, Node>;

   public:
    SimpleGraph() : _nodesNum(0), _edges(), _adjacencyList() {}
    SimpleGraph(Node nodesNum)
        : _nodesNum(nodesNum), _edges(), _adjacencyList(nodesNum) {}

    static SimpleGraph FromEdges(Node nodesNum, const std::set<Edge>& edges);
    static SimpleGraph EmptyGraph(Node nodesNum);
    static SimpleGraph GnpRandomGraph(Node nodesNum, double probability);

    bool operator==(const SimpleGraph& other) const;
    bool operator!=(const SimpleGraph& other) const;

    /// @brief Returns a random degree-regular graph on nodesNum.
    /// @param nodesNum The number of nodes. The value of (nodesNum * degree)
    /// must be even.
    /// @param degree The degree of each node.
    /// @return a random degree-regular graph on nodesNum.
    /// @throw std::invalid_argument if the value of (nodesNum * degree) is odd
    /// or degree is not in the range [0, nodesNum).
    /// @details A regular graph is a graph where each node has the same number
    /// of neighbors.
    static SimpleGraph RandomRegularGraph(Node nodesNum, Node degree);

    Node NodesNum() const;
    const std::set<Edge>& Edges() const;
    const std::vector<std::set<Node>>& AdjacencyList() const;
    void AddEdge(Edge edge);
    void AddEdge(Node u, Node v);
    void RemoveEdge(Edge edge);
    void RemoveEdge(Node u, Node v);
    void Clear();
    bool HasEdge(Edge edge) const;
    bool HasEdge(Node u, Node v) const;
    std::set<Node> Neighbors(Node n) const;
    Node Degree(Node n) const;
    Node MaxDegree() const;
    void Resize(Node nodesNum);

    /// @brief Calculate the length of the shortest path from the specified
    /// start node to the end node.
    /// @param start The index of the start node.
    /// @param end The index of the end node.
    /// @return The length of the path from the start node to the end node. If
    /// there is no path, -1 is returned.
    Node ShortestPathLength(Node start, Node end) const;

    double AverageShortestPathLength() const;
    bool TwoOpt(Edge edge1, Edge edge2);
    bool TwoOpt(Node parent1, Node child1, Node parent2, Node child2);

   private:
    Node _nodesNum;
    std::set<Edge> _edges;
    std::vector<std::set<Node>> _adjacencyList;

    template <typename... Args>
    void ValidateIndexes(Args... indexes) const;
    template <typename... Args>
    void ValidateEdges(const Args&... edges) const;
    void NormalizeEdges(Edge& edges) const;
};

}  // namespace Eacpp