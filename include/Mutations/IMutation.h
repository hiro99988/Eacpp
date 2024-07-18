#ifndef IMutation_h
#define IMutation_h

#include <eigen3/Eigen/Core>

#include "Rng/HasRng.h"
#include "Rng/IRng.h"

namespace Eacpp {

template <typename T>
struct IMutation : protected HasRng {
    IMutation() {}
    explicit IMutation(IRng* rng) : HasRng(rng) {}
    virtual ~IMutation() {}

    virtual void Mutate(Eigen::ArrayX<T>& individual) const = 0;
};

}  // namespace Eacpp

#endif