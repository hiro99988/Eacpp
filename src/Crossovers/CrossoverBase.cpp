#include "Crossovers/CrossoverBase.h"

#include <iostream>

namespace Eacpp {

template <Number T>
std::vector<Eigen::VectorX<T>> CrossoverBase<T>::Cross(std::vector<Eigen::VectorX<T>> parents) {
    int size = std::size(parents);
    if (size != _parentNum) {
        throw std::invalid_argument("Invalid number of parents");
    }
    return performCrossover(parents);
}

}  // namespace Eacpp