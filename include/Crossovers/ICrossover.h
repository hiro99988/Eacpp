#pragma once

#include <Eigen/Core>
#include <vector>

#include "Individual.h"

namespace Eacpp {

template <typename T>
struct ICrossover {
    virtual ~ICrossover() {}

    virtual int GetParentNum() const = 0;
    virtual Individual<T> Cross(
        const std::vector<Individual<T>>& parents) const = 0;
};

}  // namespace Eacpp
