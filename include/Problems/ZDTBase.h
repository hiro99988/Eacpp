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
    ZDTBase() {}
    ZDTBase(int decisionVariablesNum) : decisionVariablesNum(decisionVariablesNum) {}
    ZDTBase(int decisionVariablesNum, double gOfParetoFront)
        : decisionVariablesNum(decisionVariablesNum), gOfParetoFront(gOfParetoFront) {}

    int DecisionVariablesNum() const override {
        return decisionVariablesNum;
    }
    int ObjectivesNum() const override {
        return objectivesNum;
    }
    void ComputeObjectiveSet(Individuald& individual) const override;
    bool IsFeasible(const Individuald& individual) const override;
    std::vector<bool> EvaluateConstraints(const Individuald& individual) const override;

    std::vector<Eigen::ArrayXd> GenerateParetoFront(int pointsNum) const override;

    constexpr std::pair<double, double> VariableBound() const {
        return variableBound;
    }

   protected:
    virtual double F1(double x1) const = 0;
    virtual double G(const Eigen::ArrayXd& X) const = 0;
    virtual double F2(double f1, double g) const = 0;

   private:
    constexpr static int objectivesNum = 2;
    constexpr static std::pair<double, double> variableBound = {0.0, 1.0};
    const int decisionVariablesNum = 30;
    const double gOfParetoFront = 1.0;
};

}  // namespace Eacpp