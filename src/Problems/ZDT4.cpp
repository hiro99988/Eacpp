#include "Problems/ZDT4.h"

#include <cmath>
#include <eigen3/Eigen/Core>
#include <numbers>

namespace Eacpp {

double ZDT4::F1(double x1) const {
    return x1;
}

double ZDT4::G(const Eigen::ArrayXd& X) const {
    double sum = 0.0;
    for (auto&& x : X) {
        sum += x * x - 10.0 * std::cos(4.0 * std::numbers::pi * x);
    }

    return 1.0 + 10.0 * (static_cast<double>(DecisionVariablesNum()) - 1.0) + sum;
}

double ZDT4::F2(double f1, double g) const {
    return 1.0 - std::sqrt(f1 / g);
}

}  // namespace Eacpp
