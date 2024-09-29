#pragma once

#include <Problems/IProblem.h>

#include <eigen3/Eigen/Core>
#include <tuple>
#include <vector>

#include "Individual/Individual.h"
#include "Problems/IBenchmark.h"

namespace Eacpp {

class ZDTBase : public IProblem<double>, public IBenchmark {
   public:
    const double gOfParetoFront;

    int decisionVariablesNum;
    int objectivesNum = 2;
    std::pair<double, double> variableBound = {0.0, 1.0};

    ZDTBase(int decisionVariablesNum) : decisionVariablesNum(decisionVariablesNum), gOfParetoFront(1.0) {}
    ZDTBase(int decisionVariablesNum, double gOfParetoFront)
        : decisionVariablesNum(decisionVariablesNum), gOfParetoFront(gOfParetoFront) {}

    void ComputeObjectiveSet(Individuald& individual) const override;
    bool IsFeasible(const Individuald& individual) const override;
    std::vector<bool> EvaluateConstraints(const Individuald& individual) const override;

    std::vector<Eigen::ArrayXd> GenerateParetoFront(int pointsNum) const override;

   protected:
    virtual double F1(double x1) const = 0;
    virtual double G(const Eigen::ArrayXd& X) const = 0;
    virtual double F2(double f1, double g) const = 0;
};

}  // namespace Eacpp