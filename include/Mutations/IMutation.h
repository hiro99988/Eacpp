#pragma once

#include <eigen3/Eigen/Core>

#include "Individual.h"

namespace Eacpp {

template <typename T>
struct IMutation {
    virtual ~IMutation() {}

    virtual void Mutate(Individual<T>& individual) const = 0;
};

}  // namespace Eacpp
