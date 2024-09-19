#pragma once

#include <array>
#include <eigen3/Eigen/Core>
#include <vector>

#include "Problems/IProblem.h"
#include "Utils/Utils.h"

namespace Eacpp {

class LZBase : public IProblem<double> {
   public:
    int objectiveNum;
    std::vector<std::array<double, 2>> variableBounds;
    int n;

    LZBase(int objectiveNum, std::vector<std::array<double, 2>> variableBounds, int decisionNum)
        : objectiveNum(objectiveNum), variableBounds(variableBounds), n(decisionNum) {
        _lastBoundIndex = variableBounds.size() - 1;

        for (int i = 1; i <= objectiveNum; i++) {
            Eigen::ArrayXi tmp = Rangeea(objectiveNum + (i % objectiveNum), decisionNum, objectiveNum);
            _Jsi.push_back(tmp - 1);
            _Jsd.push_back(tmp.cast<double>());
        }
    }
    virtual ~LZBase() {}

    bool IsFeasible(const Eigen::ArrayXd& solution) const override;
    std::vector<bool> EvaluateConstraints(const Eigen::ArrayXd& solution) const override;

   protected:
    std::vector<Eigen::ArrayXi> _Jsi;
    std::vector<Eigen::ArrayXd> _Jsd;

    virtual double Beta(const Eigen::ArrayXd& solution, const int JIndex) const = 0;

   private:
    int _lastBoundIndex;
};

}  // namespace Eacpp
