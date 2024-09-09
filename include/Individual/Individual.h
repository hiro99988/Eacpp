#pragma once

#include <eigen3/Eigen/Core>

namespace Eacpp {

template <typename T>
struct Individual {
    Eigen::ArrayX<T> solution;
    Eigen::ArrayXd objectives;
};

}  // namespace Eacpp