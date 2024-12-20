#pragma once

#include <eigen3/Eigen/Core>
#include <limits>

#include "Decompositions/IDecomposition.h"

namespace Eacpp {

class DecompositionBase : public IDecomposition {
   public:
    virtual ~DecompositionBase() {}

    const Eigen::ArrayXd& IdealPoint() const override {
        return _idealPoint;
    }

    void InitializeIdealPoint(int objectivesNum) override {
        _idealPoint = Eigen::ArrayXd::Constant(
            objectivesNum, std::numeric_limits<double>::max());
    }

    void UpdateIdealPoint(const Eigen::ArrayXd& objectiveSet) override {
        if (_idealPoint.size() == 0) {
            _idealPoint = objectiveSet;
        } else {
            _idealPoint = _idealPoint.min(objectiveSet);
        }
    }

   protected:
    Eigen::ArrayXd _idealPoint;
};

}  // namespace Eacpp