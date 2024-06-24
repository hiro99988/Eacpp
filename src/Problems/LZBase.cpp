#include "Problems/LZBase.h"

#include <algorithm>

namespace Eacpp {

bool LZBase::IsFeasible(const Eigen::ArrayXd& solution) const {
    auto evaluation = EvaluateConstraints(solution);
    return std::all_of(evaluation.begin(), evaluation.end(), [](bool e) { return e == true; });
}

std::vector<bool> LZBase::EvaluateConstraints(const Eigen::ArrayXd& solution) const {
    std::vector<bool> evaluation;
    for (int i = 0; i < _lastBoundIndex; i++) {
        evaluation.push_back(solution(i) >= variableBounds[i][0] && solution(i) <= variableBounds[i][1]);
    }
    for (int i = _lastBoundIndex; i < solution.size(); i++) {
        evaluation.push_back(solution(i) >= variableBounds[i][0] && solution(i) <= variableBounds[i][1]);
    }
    return evaluation;
}

}  // namespace Eacpp