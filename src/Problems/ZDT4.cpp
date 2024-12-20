#include "Problems/ZDT4.h"

#include <cmath>
#include <eigen3/Eigen/Core>
#include <numbers>

#include "Individual.h"

namespace Eacpp {

double ZDT4::F1(double x1) const {
    return x1;
}

double ZDT4::G(const Eigen::ArrayXd& X) const {
    double sum = 0.0;
    for (auto&& x : X) {
        sum += x * x - 10.0 * std::cos(4.0 * std::numbers::pi * x);
    }

    return 1.0 + 10.0 * (static_cast<double>(DecisionVariablesNum()) - 1.0) +
           sum;
}

double ZDT4::H(double f1, double g) const {
    return 1.0 - std::sqrt(f1 / g);
}

std::vector<bool> ZDT4::EvaluateConstraints(
    const Individuald& individual) const {
    std::vector<bool> evaluation(individual.solution.size());
    evaluation[0] = individual.solution(0) >= VariableBounds()[0].first &&
                    individual.solution(0) <= VariableBounds()[0].second;
    for (int i = 1; i < individual.solution.size(); i++) {
        evaluation[i] = individual.solution(i) >= VariableBounds()[1].first &&
                        individual.solution(i) <= VariableBounds()[1].second;
    }

    return evaluation;
}

}  // namespace Eacpp
