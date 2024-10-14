#pragma once

#include <eigen3/Eigen/Core>
#include <tuple>
#include <vector>

#include "Individual/Individual.h"
#include "Problems/IBenchmark.h"
#include "Problems/IProblem.h"

namespace Eacpp {

class ZDTBase : public IProblem<double>, public IBenchmark {
   public:
    ZDTBase() {}
    ZDTBase(int decisionVariablesNum) : decisionVariablesNum(decisionVariablesNum) {}
    ZDTBase(int decisionVariablesNum, double gOfParetoFront)
        : decisionVariablesNum(decisionVariablesNum), gOfParetoFront(gOfParetoFront) {}
    ZDTBase(int decisionVariablesNum, const std::vector<std::pair<double, double>>& variableBounds)
        : decisionVariablesNum(decisionVariablesNum), variableBounds(variableBounds) {}
    ZDTBase(int decisionVariablesNum, double gOfParetoFront, const std::vector<std::pair<double, double>>& variableBounds)
        : decisionVariablesNum(decisionVariablesNum), gOfParetoFront(gOfParetoFront), variableBounds(variableBounds) {}
    virtual ~ZDTBase() {}

    int DecisionVariablesNum() const override {
        return decisionVariablesNum;
    }
    int ObjectivesNum() const override {
        return objectivesNum;
    }
    const std::vector<std::pair<double, double>>& VariableBounds() const override {
        return variableBounds;
    }
    void ComputeObjectiveSet(Individuald& individual) const override;
    bool IsFeasible(const Individuald& individual) const override;
    std::vector<bool> EvaluateConstraints(const Individuald& individual) const override;

    std::vector<Eigen::ArrayXd> GenerateParetoFront(int pointsNum) const override;

   protected:
    virtual double F1(double x1) const = 0;
    virtual double G(const Eigen::ArrayXd& X) const = 0;
    virtual double F2(double f1, double g) const = 0;

   private:
    constexpr static int objectivesNum = 2;
    const std::vector<std::pair<double, double>> variableBounds = {{0.0, 1.0}};
    const int decisionVariablesNum = 30;
    const double gOfParetoFront = 1.0;
};

}  // namespace Eacpp