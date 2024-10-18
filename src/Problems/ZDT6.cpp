#include "Problems/ZDT6.h"

#include <cmath>
#include <eigen3/Eigen/Core>
#include <numbers>

#include "Utils/Utils.h"

namespace Eacpp {

double ZDT6::F1(double x1) const {
    return 1.0 - std::exp(-4.0 * x1) * std::pow(std::sin(6.0 * std::numbers::pi * x1), 6.0);
}

double ZDT6::G(const Eigen::ArrayXd& X) const {
    double sum = X.sum();
    return 1.0 + 9.0 * std::pow(sum / (static_cast<double>(DecisionVariablesNum()) - 1.0), 0.25);
}

double ZDT6::H(double f1, double g) const {
    return 1.0 - std::pow(f1 / g, 2.0);
}

std::vector<Eigen::ArrayXd> ZDT6::GenerateParetoFront(int pointsNum) const {
    constexpr std::pair<double, double> region = {0.2807753191, 1.0};

    std::vector<double> x = Ranged(region.first, region.second, (region.second - region.first) / (pointsNum - 1));

    std::vector<Eigen::ArrayXd> result(pointsNum, Eigen::ArrayXd(ObjectivesNum()));
    for (int i = 0; i < pointsNum; i++) {
        double h = H(F1(x[i]), GOfParetoFront());
        double f2 = GOfParetoFront() * h;
        result[i] << x[i], f2;
    }
    return result;
}

}  // namespace Eacpp
