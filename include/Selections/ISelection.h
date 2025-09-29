#pragma once

#include <vector>

namespace eacpp {

struct ISelection {
    virtual ~ISelection() {}

    virtual std::vector<int> Select(
        int parentNum, const std::vector<int>& population) const = 0;
};

}  // namespace eacpp
