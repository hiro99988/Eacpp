#include "Crossovers/OnePointCrossover.h"

#include "Utils/TemplateType.h"

namespace Eacpp {

template <Number T>
Eigen::ArrayXX<T> OnePointCrossover<T>::performCrossover(const Eigen::ArrayXX<T>& parents) const {
    int size = parents[0].size();
    int crossoverPoint = _rng.Integer(1, size - 1);
    Eigen::ArrayX<T> child(size);
    child << parents[0].head(crossoverPoint), parents[1].tail(size - crossoverPoint);
    return {child};
}

}  // namespace Eacpp