#include "Problems/ZDT6.h"

#include <cmath>
#include <eigen3/Eigen/Core>
#include <numbers>

namespace Eacpp {

double ZDT6::F1(double x1) const {
    return 1.0 - std::exp(-4.0 * x1) * std::pow(std::sin(6.0 * std::numbers::pi * x1), 6.0);
}

double ZDT6::G(const Eigen::ArrayXd& X) const {
    double sum = X.sum();
    return 1.0 + 9.0 * std::pow(sum / (static_cast<double>(DecisionVariablesNum()) - 1.0), 0.25);
}

double ZDT6::F2(double f1, double g) const {
    return 1.0 - std::pow(f1 / g, 2.0);
}

}  // namespace Eacpp
