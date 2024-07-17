#include "Problems/LZ1.h"

#include <eigen3/Eigen/Dense>
#include <iostream>

namespace Eacpp {

Eigen::ArrayXd LZ1::ComputeObjectiveSet(const Eigen::ArrayXd& solution) const {
    Eigen::ArrayXd objectiveSet(objectiveNum);
    double ob1 = F1(solution);
    double ob2 = F2(solution);
    objectiveSet << ob1, ob2;
    return objectiveSet;
}

double LZ1::F1(const Eigen::ArrayXd& solution) const {
    double beta = Beta(solution, 0);
    return solution(0) + beta;
}

double LZ1::F2(const Eigen::ArrayXd& solution) const {
    double beta = Beta(solution, 1);
    return 1.0 - std::sqrt(solution(0)) + beta;
}

double LZ1::Beta(const Eigen::ArrayXd& solution, const int JIndex) const {
    Eigen::ArrayXd g = G(solution, _Jsd[JIndex]);
    Eigen::ArrayXd diff = solution(_Jsi[JIndex]) - g;
    double sum = diff.pow(2).sum();
    return 2.0 / _Jsd[JIndex].size() * sum;
}

Eigen::ArrayXd LZ1::G(const Eigen::ArrayXd& solution, const Eigen::ArrayXd& J) const {
    Eigen::ArrayXd power = 0.5 * (1.0 + 3.0 * (J - 2.0) / (n - 2.0));
    Eigen::ArrayXd g = Eigen::pow(solution(0), power);
    return g;
}

}  // namespace Eacpp