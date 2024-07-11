#include "Problems/LZ1.h"

#include <eigen3/Eigen/Dense>
#include <iostream>

namespace Eacpp {

Eigen::ArrayXd LZ1::ComputeObjectiveSet(const Eigen::ArrayXd& solution) const {
    Eigen::ArrayXd objectiveSet(objectiveNum);
    double ob1 = f1(solution);
    double ob2 = f2(solution);
    objectiveSet << ob1, ob2;
    return objectiveSet;
}

double LZ1::f1(const Eigen::ArrayXd& solution) const {
    double b = beta(solution, _Js[0]);
    return solution(0) + b;
}

double LZ1::f2(const Eigen::ArrayXd& solution) const {
    double b = beta(solution, _Js[1]);
    return 1.0 - std::sqrt(solution(0)) + b;
}

double LZ1::beta(const Eigen::ArrayXd& solution, const Eigen::ArrayXd& J) const {
    int size = J.size();
    Eigen::ArrayXd g = g(solution, J);
    Eigen::ArrayXd diff(size);
    for (int i = 0; i < size; i++) {
        diff(i) = solution(static_cast<int>(J(i))) - g(i);
    }
    return diff.pow(2).sum();
}

Eigen::ArrayXd LZ1::g(const Eigen::ArrayXd& solution, const Eigen::ArrayXd& J) const {
    Eigen::ArrayXd power = 0.5 * (1.0 + 3.0 * (J - 2.0) / (n - 2.0));
    Eigen::ArrayXd result = Eigen::pow(solution(0), power);
    return result;
}

}  // namespace Eacpp