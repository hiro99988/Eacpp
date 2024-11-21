#include "Samplings/RealRandomSampling.h"

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <tuple>
#include <vector>

#include "Individual.h"

namespace Eacpp {

std::vector<Individuald> RealRandomSampling::Sample(const int sampleNum, const int variableNum) const {
    std::vector<Eigen::ArrayXd> solutions;
    std::vector<Individuald> individuals;
    individuals.reserve(sampleNum);

    if (variableBounds.size() == 1) {
        solutions = _rng->Uniform(variableBounds[0].first, variableBounds[0].second, {sampleNum, variableNum});
        for (const auto& solution : solutions) {
            individuals.emplace_back(solution);
        }
    } else {
        for (int i = 0; i < sampleNum; i++) {
            Eigen::ArrayXd solution(variableNum);
            for (int j = 0; j < variableNum; j++) {
                if (j < variableBounds.size()) {
                    solution(j) = _rng->Uniform(variableBounds[j].first, variableBounds[j].second);
                } else {
                    solution(j) = _rng->Uniform(variableBounds.back().first, variableBounds.back().second);
                }
            }
            individuals.emplace_back(solution);
        }
    }

    return individuals;
}

}  // namespace Eacpp