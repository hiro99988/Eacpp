#ifndef IMutation_h
#define IMutation_h

#include <eigen3/Eigen/Core>

#include "Rng/IRng.h"
#include "Rng/Rng.h"

namespace Eacpp {

template <typename T>
struct IMutation {
    virtual ~IMutation() {}

    virtual void Mutate(Eigen::ArrayX<T>& individual) const = 0;
};

}  // namespace Eacpp

#endif