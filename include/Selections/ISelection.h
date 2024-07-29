#ifndef ISelection_h
#define ISelection_h

#include <vector>

#include "Utils/TemplateType.h"

namespace Eacpp {

struct ISelection {
    virtual ~ISelection() {}

    virtual std::vector<int> Select(int parentNum, const std::vector<int>& population) const = 0;
};

}  // namespace Eacpp

#endif