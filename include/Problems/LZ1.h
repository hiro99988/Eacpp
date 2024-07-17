#ifndef LZ1_h
#define LZ1_h

#include <eigen3/Eigen/Core>

#include "Problems/LZBase.h"

namespace Eacpp {

class LZ1 : public LZBase {
   public:
    LZ1(int decisionNum) : LZBase(2, {{0, 1}}, decisionNum) {}

    Eigen::ArrayXd ComputeObjectiveSet(const Eigen::ArrayXd& solution) const override;

   private:
    double F1(const Eigen::ArrayXd& solution) const;
    double F2(const Eigen::ArrayXd& solution) const;
    double Beta(const Eigen::ArrayXd& solution, const int JIndex) const override;
    Eigen::ArrayXd G(const Eigen::ArrayXd& solution, const Eigen::ArrayXd& J) const;
};

}  // namespace Eacpp

#endif