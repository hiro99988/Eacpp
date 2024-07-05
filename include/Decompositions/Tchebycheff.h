#ifndef Tchebycheff_h
#define Tchebycheff_h

#include "Decompositions/IDecomposition.h"

namespace Eacpp {

class Tchebycheff : public IDecomposition {
   public:
    double ComputeObjective(const Eigen::ArrayXd& weight, const Eigen::ArrayXd& objectiveSet,
                            const Eigen::ArrayXd& referencePoint) const override;
};

}  // namespace Eacpp

#endif