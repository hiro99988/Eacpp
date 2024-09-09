#ifndef ISampling_h
#define ISampling_h

#include <eigen3/Eigen/Core>
#include <vector>

namespace Eacpp {

template <typename T>
struct ISampling {
    virtual ~ISampling() {}

    virtual std::vector<Eigen::ArrayX<T>> Sample(const int sampleNum, const int variableNum) const = 0;
};

}  // namespace Eacpp

#endif