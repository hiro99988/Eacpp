#ifndef IDeomposition_h
#define IDeomposition_h

#include <eigen3/Eigen/Core>
namespace Eacpp {

struct IDecomposition {
    virtual double ComputeObjective(const Eigen::ArrayXd& weight, const Eigen::ArrayXd& objectiveSet,
                                    const Eigen::ArrayXd& referencePoint) const = 0;
};

}  // namespace Eacpp

#endif