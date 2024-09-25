#include "Problems/ZDTBase.h"

#include <algorithm>
#include <eigen3/Eigen/Core>
#include <vector>

#include "Individual/Individual.h"

namespace Eacpp {

void ZDTBase::ComputeObjectiveSet(Individuald& individual) const {
    double f1 = F1(individual.solution[0]);
    double g = G(individual.solution.tail(individual.solution.size() - 1));
    double f2 = F2(f1, g);
    Eigen::ArrayXd result(objectivesNum);
    result << f1, f2;

    individual.objectives = result;
}

bool ZDTBase::IsFeasible(const Individuald& individual) const {
    auto evaluation = EvaluateConstraints(individual);
    return std::all_of(evaluation.begin(), evaluation.end(), [](bool e) { return e == true; });
}

std::vector<bool> ZDTBase::EvaluateConstraints(const Individuald& individual) const {
    std::vector<bool> evaluation(individual.solution.size());
    for (int i = 0; i < individual.solution.size(); i++) {
        evaluation[i] = individual.solution(i) >= variableBound.first && individual.solution(i) <= variableBound.second;
    }
    return evaluation;
}

}  // namespace Eacpp