#ifndef ISampling_h
#define ISampling_h

#include <eigen3/Eigen/Core>

#include "Utils/TemplateType.h"

namespace Eacpp {

template <Number T>
struct ISampling {
    virtual ~ISampling() {}

    virtual Eigen::ArrayXX<T> Sample(int sampleNum, int variableNum) const = 0;
};

}  // namespace Eacpp

#endif