#pragma once

#include <eigen3/Eigen/Core>
#include <limits>

#include "Decompositions/IDecomposition.h"

namespace Eacpp {

class DecompositionBase : public IDecomposition {
   public:
    DecompositionBase(int objectivesNum) {
        _idealPoint = Eigen::ArrayXd::Constant(objectivesNum, std::numeric_limits<double>::max());
    }
    virtual ~DecompositionBase() {}

    Eigen::ArrayXd& IdealPoint() override { return _idealPoint; }

    void UpdateIdealPoint(const Eigen::ArrayXd& objectiveSet) override { _idealPoint = _idealPoint.min(objectiveSet); }

   protected:
    Eigen::ArrayXd _idealPoint;
};

}  // namespace Eacpp