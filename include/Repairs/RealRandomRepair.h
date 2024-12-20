#pragma once

#include <memory>
#include <tuple>
#include <vector>

#include "Individual.h"
#include "Problems/IProblem.h"
#include "Repairs/IRepair.h"
#include "Rng/IRng.h"
#include "Rng/Rng.h"

namespace Eacpp {

class RealRandomRepair : public IRepair<double> {
   public:
    RealRandomRepair(const std::shared_ptr<IProblem<double>>& problem)
        : _problem(problem) {
        _rng = std::make_shared<Rng>();
    }
    RealRandomRepair(const std::shared_ptr<IProblem<double>>& problem,
                     const std::shared_ptr<IRng>& rng)
        : _problem(problem), _rng(rng) {}

    void Repair(Individuald& individual) override {
        std::vector<bool> evaluation =
            _problem->EvaluateConstraints(individual);
        int variableBoundsSize = _problem->VariableBounds().size();
        for (int i = 0; i < evaluation.size(); i++) {
            if (evaluation[i]) {
                continue;
            }

            if (i < variableBoundsSize) {
                individual.solution(i) =
                    _rng->Uniform(_problem->VariableBounds()[i].first,
                                  _problem->VariableBounds()[i].second);
            } else {
                individual.solution(i) =
                    _rng->Uniform(_problem->VariableBounds().back().first,
                                  _problem->VariableBounds().back().second);
            }
        }
    }

   private:
    std::shared_ptr<IProblem<double>> _problem;
    std::shared_ptr<IRng> _rng;
};

}  // namespace Eacpp
