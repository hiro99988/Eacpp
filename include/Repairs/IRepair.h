#pragma once

#include "Individual.h"

namespace Eacpp {

template <typename T>
struct IRepair {
    virtual ~IRepair() {}

    virtual void Repair(Individual<T>& individual) = 0;
};

}  // namespace Eacpp
