#include "Problems/ZDT1.h"

#include <cmath>
#include <eigen3/Eigen/Core>

namespace Eacpp {

double ZDT1::F1(double x1) const {
    return x1;
}

double ZDT1::G(const Eigen::ArrayXd& X) const {
    double sum = X.sum();
    return 1.0 + 9.0 * sum / (static_cast<double>(DecisionVariablesNum() - 1));
}

double ZDT1::H(double f1, double g) const {
    return 1.0 - std::sqrt(f1 / g);
}

}  // namespace Eacpp
