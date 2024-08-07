#include "Problems/ZDTBase.h"

#include <algorithm>
#include <eigen3/Eigen/Core>
#include <vector>

namespace Eacpp {

Eigen::ArrayXd ZDTBase::ComputeObjectiveSet(const Eigen::ArrayXd& solution) const {
    double f1 = F1(solution[0]);
    double g = G(solution.tail(solution.size() - 1));
    double f2 = F2(f1, g);
    Eigen::ArrayXd result(2);
    result << f1, f2;
    return result;
}

bool ZDTBase::IsFeasible(const Eigen::ArrayXd& solution) const {
    auto evaluation = EvaluateConstraints(solution);
    return std::all_of(evaluation.begin(), evaluation.end(), [](bool e) { return e == true; });
}

std::vector<bool> ZDTBase::EvaluateConstraints(const Eigen::ArrayXd& solution) const {
    std::vector<bool> evaluation(solution.size());
    for (int i = 0; i < solution.size(); i++) {
        evaluation[i] = solution(i) >= variableBound[0] && solution(i) <= variableBound[1];
    }
    return evaluation;
}

}  // namespace Eacpp