#pragma once

#include <pybind11/embed.h>

#include "Individual.h"
#include "Problems/IProblem.h"

namespace Eacpp {

class Hpa : public IProblem<double> {
   private:
    int _decisionVariablesNum;
    int _objectivesNum;
    const std::vector<std::pair<double, double>> _variableBounds = {{0.0, 1.0}};
    pybind11::object _hpa;

   public:
    Hpa(const pybind11::module& module, const char* problemName, int n_div = 4,
        int level = 0, bool normalize = true);
    ~Hpa() {}

    int DecisionVariablesNum() const override;
    int ObjectivesNum() const override;
    const std::vector<std::pair<double, double>>& VariableBounds()
        const override;
    void ComputeObjectiveSet(Individuald& individual) const override;
    bool IsFeasible(const Individuald& individual) const override;
    std::vector<bool> EvaluateConstraints(
        const Individuald& individual) const override;
};

}  // namespace Eacpp
