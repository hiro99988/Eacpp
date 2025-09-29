#pragma once

#include "Individual.h"

namespace eacpp {

template <typename T>
struct IRepair {
    virtual ~IRepair() {}

    virtual void Repair(Individual<T>& individual) = 0;
};

}  // namespace eacpp
