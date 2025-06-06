#include "Problems/ZDTBase.h"

#include <Eigen/Core>
#include <algorithm>
#include <vector>

#include "Individual.h"
#include "Utils/Utils.h"

namespace Eacpp {

void ZDTBase::ComputeObjectiveSet(Individuald& individual) const {
    double f1 = F1(individual.solution[0]);
    double g = G(individual.solution.tail(individual.solution.size() - 1));
    double h = H(f1, g);
    double f2 = g * h;
    Eigen::ArrayXd result(objectivesNum);
    result << f1, f2;

    individual.objectives = result;
}

bool ZDTBase::IsFeasible(const Individuald& individual) const {
    auto evaluation = EvaluateConstraints(individual);
    return std::all_of(evaluation.begin(), evaluation.end(),
                       [](bool e) { return e == true; });
}

std::vector<bool> ZDTBase::EvaluateConstraints(
    const Individuald& individual) const {
    std::vector<bool> evaluation(individual.solution.size());
    for (int i = 0; i < individual.solution.size(); i++) {
        evaluation[i] = individual.solution(i) >= variableBounds[0].first &&
                        individual.solution(i) <= variableBounds[0].second;
    }
    return evaluation;
}

std::vector<Eigen::ArrayXd> ZDTBase::GenerateParetoFront(int pointsNum) const {
    std::vector<double> x =
        LinSpace(variableBounds[0].first, variableBounds[0].second, pointsNum);
    std::vector<Eigen::ArrayXd> result(pointsNum,
                                       Eigen::ArrayXd(objectivesNum));
    for (int i = 0; i < pointsNum; i++) {
        double h = H(x[i], gOfParetoFront);
        double f2 = gOfParetoFront * h;
        result[i] << x[i], f2;
    }
    return result;
}

}  // namespace Eacpp