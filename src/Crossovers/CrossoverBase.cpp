#include "Crossovers/CrossoverBase.h"

#include <iostream>

namespace Eacpp {

template <Number T>
Eigen::ArrayXX<T> CrossoverBase<T>::Cross(const Eigen::ArrayXX<T>& parents) const {
    int size = parents.size();
    if (size != _parentNum) {
        throw std::invalid_argument("Invalid number of parents");
    }
    return performCrossover(parents);
}

}  // namespace Eacpp