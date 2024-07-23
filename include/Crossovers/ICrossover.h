#ifndef ICrossover_H
#define ICrossover_H

#include <eigen3/Eigen/Core>
#include <vector>

#include "Utils/TemplateType.h"

namespace Eacpp {

template <Number T>
struct ICrossover {
    virtual ~ICrossover() {}

    virtual int GetParentNum() const = 0;
    virtual Eigen::ArrayX<T> Cross(const std::vector<Eigen::ArrayX<T>>& parents) const = 0;
};

}  // namespace Eacpp

#endif