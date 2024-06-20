#include "Crossovers/OnePointCrossover.h"

#include "Utils/TemplateType.h"

namespace Eacpp {

template <Number T>
std::vector<Eigen::VectorX<T>> OnePointCrossover<T>::performCrossover(std::vector<Eigen::VectorX<T>> parents) {
    int size = parents[0].size();
    // int crossoverPoint = _rng.Integer(1, size - 1);
    int crossoverPoint = 1;
    Eigen::VectorX<T> child(size);
    child << parents[0].head(crossoverPoint), parents[1].tail(size - crossoverPoint);
    return {child};
}

}  // namespace Eacpp