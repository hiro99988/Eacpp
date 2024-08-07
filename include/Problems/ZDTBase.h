#pragma once

#include <Problems/IProblem.h>

#include <array>
#include <eigen3/Eigen/Core>
#include <vector>

namespace Eacpp {

class ZDTBase : public IProblem<double> {
   public:
    int decisionNum;
    int objectiveNum = 2;
    std::array<double, 2> variableBound = {0.0, 1.0};

    ZDTBase(int decisionVariableNum) : decisionNum(decisionVariableNum) {}

    Eigen::ArrayXd ComputeObjectiveSet(const Eigen::ArrayXd& solution) const override;
    bool IsFeasible(const Eigen::ArrayXd& solution) const override;
    std::vector<bool> EvaluateConstraints(const Eigen::ArrayXd& solution) const override;

    virtual double F1(double x1) const = 0;
    virtual double G(const Eigen::ArrayXd& X) const = 0;
    virtual double F2(double f1, double g) const = 0;
};

}  // namespace Eacpp