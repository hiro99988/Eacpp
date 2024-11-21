#pragma once

#include <stdexcept>
#include <vector>

namespace Eacpp {

template <typename T>
class SimpleGraph {
   public:
    SimpleGraph(int nodesNum) : nodesNum(nodesNum), matrix(ElementsNum(nodesNum), 0) {}
    SimpleGraph(int nodesNum, const std::vector<T>& matrix) : nodesNum(nodesNum), matrix(matrix) {
        if (matrix.size() != ElementsNum(nodesNum)) {
            throw std::invalid_argument("nodesNum and matrix size mismatch");
        }
    }

    typename std::vector<T>::reference operator()(int row, int col);
    typename std::vector<T>::const_reference operator()(int row, int col) const;

    const std::vector<T>& Matrix() const {
        return matrix;
    }

    /// @brief Calculate the length of the shortest path from the specified start node to the end node.
    /// @param start The index of the start node.
    /// @param end The index of the end node.
    /// @return The length of the path from the start node to the end node. If there is no path, -1 is returned.
    int ShortestPathLength(int start, int end) const;

    int MaxDegree() const;
    int AverageShortestPathLength() const;
    std::vector<std::vector<int>> ToAdjacencyList() const;

   private:
    int nodesNum;
    std::vector<T> matrix;

    size_t Index(size_t row, size_t col) const;
    size_t ElementsNum(int nodesNum) const;
    T Element(size_t row, size_t col) const;
};

template class SimpleGraph<bool>;
template class SimpleGraph<int>;
template class SimpleGraph<unsigned int>;

}  // namespace Eacpp