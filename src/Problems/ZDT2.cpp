#include "Problems/ZDT2.h"

#include <cmath>
#include <eigen3/Eigen/Core>

namespace Eacpp {

double ZDT2::F1(double x1) const { return x1; }

double ZDT2::G(const Eigen::ArrayXd& X) const {
    double sum = X.sum();
    return 1.0 + 9.0 * sum / (decisionVariablesNum - 1);
}

double ZDT2::F2(double f1, double g) const { return 1.0 - std::pow(f1 / g, 2); }

}  // namespace Eacpp
