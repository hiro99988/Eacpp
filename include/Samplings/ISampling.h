#ifndef ISampling_h
#define ISampling_h

#include <eigen3/Eigen/Core>
#include <vector>

#include "Utils/TemplateType.h"

namespace Eacpp {

template <Number T>
struct ISampling {
    virtual ~ISampling() {}

    virtual std::vector<Eigen::ArrayX<T>> Sample(const int sampleNum, const int variableNum) const = 0;
};

}  // namespace Eacpp

#endif