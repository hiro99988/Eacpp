#include "Problems/ZDT3.h"

#include <array>
#include <cmath>
#include <eigen3/Eigen/Core>
#include <numbers>
#include <vector>

#include "Utils/Utils.h"

namespace Eacpp {

double ZDT3::F1(double x1) const {
    return x1;
}

double ZDT3::G(const Eigen::ArrayXd& X) const {
    double sum = X.sum();
    return 1.0 + 9.0 * sum / (DecisionVariablesNum() - 1);
}

double ZDT3::H(double f1, double g) const {
    double div = f1 / g;
    return 1.0 - std::sqrt(div) - div * std::sin(10.0 * std::numbers::pi * f1);
}

std::vector<Eigen::ArrayXd> ZDT3::GenerateParetoFront(int pointsNum) const {
    constexpr int regionsSize = 5;
    constexpr std::array<std::pair<double, double>, regionsSize> regions = {{{0.0, 0.0830015349},
                                                                             {0.182228780, 0.2577623634},
                                                                             {0.4093136748, 0.4538821041},
                                                                             {0.6183967944, 0.6525117038},
                                                                             {0.8233317983, 0.8518328654}}};

    int pointsNumPerRegion = pointsNum / regionsSize;
    std::vector<double> xs;
    xs.reserve(pointsNum);
    for (auto&& r : regions) {
        std::vector<double> x = Ranged(r.first, r.second, (r.second - r.first) / (pointsNumPerRegion - 1));
        xs.insert(xs.end(), std::make_move_iterator(x.begin()), std::make_move_iterator(x.end()));
    }

    std::vector<Eigen::ArrayXd> result(pointsNum, Eigen::ArrayXd(ObjectivesNum()));
    for (int i = 0; i < xs.size(); i++) {
        double f1 = F1(xs[i]);
        double h = H(f1, GOfParetoFront());
        double f2 = GOfParetoFront() * h;
        result[i] << f1, f2;
    }

    return result;
}

}  // namespace Eacpp
