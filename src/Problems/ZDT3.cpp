#include "Problems/ZDT3.h"

#include <cmath>
#include <eigen3/Eigen/Core>
#include <numbers>

namespace Eacpp {

double ZDT3::F1(double x1) const {
    return x1;
}

double ZDT3::G(const Eigen::ArrayXd& X) const {
    double sum = X.sum();
    return 1.0 + 9.0 * sum / (DecisionVariablesNum() - 1);
}

double ZDT3::F2(double f1, double g) const {
    double div = f1 / g;
    return 1.0 - std::sqrt(div) - div * std::sin(10.0 * std::numbers::pi * f1);
}

}  // namespace Eacpp
