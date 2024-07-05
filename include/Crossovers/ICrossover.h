#ifndef ICrossover_H
#define ICrossover_H

#include <eigen3/Eigen/Core>

#include "Utils/TemplateType.h"

namespace Eacpp {

template <Number T>
struct ICrossover {
    virtual ~ICrossover() {}

    virtual int GetParentNum() const = 0;
    virtual int GetChildrenNum() const = 0;
    virtual Eigen::ArrayXX<T> Cross(const Eigen::ArrayXX<T>& parents) const = 0;
};

}  // namespace Eacpp

#endif