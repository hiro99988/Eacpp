#include "Selections/RandomSelection.h"

namespace Eacpp {

template <Number T>
Eigen::ArrayXX<T> RandomSelection<T>::select(int parentNum, const Eigen::ArrayXX<T>& population) const {
    return _rng.Choice(population, parentNum, false);
}

}  // namespace Eacpp