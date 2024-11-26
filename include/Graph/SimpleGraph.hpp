#pragma once

#include <concepts>
#include <stdexcept>
#include <vector>

namespace Eacpp {

template <typename T>
class SimpleGraph {
   public:
    SimpleGraph() : _nodesNum(0), _matrix() {}
    SimpleGraph(int nodesNum) : _nodesNum(nodesNum), _matrix(ElementsNum(nodesNum), 0) {}
    SimpleGraph(int nodesNum, const std::vector<T>& matrix) : _nodesNum(nodesNum), _matrix(matrix) {
        if (matrix.size() != ElementsNum(nodesNum)) {
            throw std::invalid_argument("nodesNum and matrix size mismatch");
        }
    }

    static size_t ElementsNum(int nodesNum);
    static SimpleGraph<T> GnpRandomGraph(int nodesNum, double probability);

    typename std::vector<T>::reference operator()(int row, int col);
    typename std::vector<T>::const_reference operator()(int row, int col) const;

    typename std::vector<T>::reference operator[](size_t index);
    typename std::vector<T>::const_reference operator[](size_t index) const;
    typename std::vector<T>::reference operator()(size_t row, size_t col);
    typename std::vector<T>::const_reference operator()(size_t row, size_t col) const;

    int size() const {
        return _matrix.size();
    }

    int NodesNum() const {
        return _nodesNum;
    }

    void resize(int nodesNum);

    /// @brief Calculate the length of the shortest path from the specified start node to the end node.
    /// @param start The index of the start node.
    /// @param end The index of the end node.
    /// @return The length of the path from the start node to the end node. If there is no path, -1 is returned.
    int ShortestPathLength(int start, int end) const;

    int MaxDegree() const;
    double AverageShortestPathLength() const;
    std::vector<std::vector<int>> ToAdjacencyList() const;
    void TwoOpt(size_t parent1, size_t parent2, size_t child1, size_t child2);
    std::vector<int> GetEdges(size_t node) const;

   private:
    int _nodesNum;
    std::vector<T> _matrix;

    size_t Index(size_t row, size_t col) const;
    T Element(size_t row, size_t col) const;

    template <typename... Args>
    void ValidateIndexes(Args... indexes) const;

    void ValidateEdge(size_t row, size_t col) const;
};  // namespace Eacpp

template class SimpleGraph<bool>;
template class SimpleGraph<int>;
template class SimpleGraph<unsigned int>;

}  // namespace Eacpp