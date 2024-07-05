#ifndef LZBase_h
#define LZBase_h

#include <array>
#include <eigen3/Eigen/Core>
#include <vector>

#include "Problems/IProblem.h"

namespace Eacpp {

class LZBase : public IProblem {
   public:
    int objectiveNum;
    std::vector<std::array<double, 2>> variableBounds;
    int n;

    LZBase(int objectiveNum, std::vector<std::array<double, 2>> variableBounds, int decisionNum)
        : objectiveNum(objectiveNum), n(decisionNum) {
        _lastBoundIndex = variableBounds.size() - 1;
        n = decisionNum;

        for (int i = 0; i < objectiveNum; i++) {
            Js.push_back(Eigen::ArrayXi::LinSpaced(int(objectiveNum / 2) - (1 - i), i + 1, decisionNum - 1));
        }
    }
    virtual ~LZBase() {}

    bool IsFeasible(const Eigen::ArrayXd& solution) const override;
    std::vector<bool> EvaluateConstraints(const Eigen::ArrayXd& solution) const override;

   protected:
    std::vector<Eigen::ArrayXd> Js;

    virtual double beta(const Eigen::ArrayXd& solution, const Eigen::ArrayXd& J) const = 0;

   private:
    int _lastBoundIndex;
};

}  // namespace Eacpp

#endif