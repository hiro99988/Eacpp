#ifndef ICrossover_H
#define ICrossover_H

#include <eigen3/Eigen/Core>
#include <vector>

#include "Utils/TemplateType.h"

namespace Eacpp {

template <Number T>
struct ICrossover {
    virtual int GetParentNum() = 0;
    virtual int GetChildrenNum() = 0;
    virtual std::vector<Eigen::VectorX<T>> Cross(std::vector<Eigen::VectorX<T>> parents) = 0;
};

}  // namespace Eacpp

#endif