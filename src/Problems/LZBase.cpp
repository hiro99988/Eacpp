#include "Problems/LZBase.h"

#include <algorithm>

namespace Eacpp {

bool LZBase::IsFeasible(const Eigen::ArrayXd& solution) const {
    auto evaluation = EvaluateConstraints(solution);
    return std::all_of(evaluation.begin(), evaluation.end(), [](bool e) { return e == true; });
}

std::vector<bool> LZBase::EvaluateConstraints(const Eigen::ArrayXd& solution) const {
    std::vector<bool> evaluation(solution.size());
    int index = 0;
    for (int i = 0; i < _lastBoundIndex; i++) {
        evaluation[index] = solution(i) >= variableBounds[i][0] && solution(i) <= variableBounds[i][1];
        ++index;
    }
    for (int i = _lastBoundIndex; i < solution.size(); i++) {
        evaluation[i] = solution(i) >= variableBounds[_lastBoundIndex][0] && solution(i) <= variableBounds[_lastBoundIndex][1];
        ++index;
    }
    return evaluation;
}

}  // namespace Eacpp