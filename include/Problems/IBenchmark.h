#pragma once

#include <Eigen/Core>
#include <vector>

namespace eacpp {

struct IBenchmark {
    virtual ~IBenchmark() {}

    virtual std::vector<Eigen::ArrayXd> GenerateParetoFront(
        int pointsNum) const = 0;
};

}  // namespace eacpp