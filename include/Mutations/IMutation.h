#pragma once

#include <Eigen/Core>

#include "Individual.h"

namespace eacpp {

template <typename T>
struct IMutation {
    virtual ~IMutation() {}

    virtual void Mutate(Individual<T>& individual) const = 0;
};

}  // namespace eacpp
