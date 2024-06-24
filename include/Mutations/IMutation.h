#ifndef IMutation_h
#define IMutation_h

#include <eigen3/Eigen/Core>

namespace Eacpp {

template <typename T>
struct IMutation {
    virtual void Mutate(Eigen::ArrayX<T>& individual) const = 0;
};

}  // namespace Eacpp

#endif