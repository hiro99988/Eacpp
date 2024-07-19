#ifndef ISelection_h
#define ISelection_h

#include <eigen3/Eigen/Core>

#include "Utils/TemplateType.h"

namespace Eacpp {

template <Number T>
struct ISelection {
    virtual ~ISelection() {}

    virtual Eigen::ArrayXX<T> Select(int parentNum, const Eigen::ArrayXX<T>& population) const = 0;
};

}  // namespace Eacpp

#endif