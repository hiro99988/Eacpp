#pragma once

#include <eigen3/Eigen/Core>

#include "Problems/ZDTBase.h"
#include "Reflections/Reflection.h"

namespace Eacpp {

class ZDT1 : public ZDTBase {
   public:
    ZDT1() : ZDTBase() {}
    ZDT1(int decisionVariablesNum) : ZDTBase(decisionVariablesNum) {}

   private:
    double F1(double x1) const override;
    double G(const Eigen::ArrayXd& x) const override;
    double H(double f1, double g) const override;

#ifdef _TEST_
   public:
    friend class ZDT1Test;
#endif
};

REGISTER_REFLECTION(IBenchmark, ZDT1, zdt1)
REGISTER_TEMPLATE_REFLECTION(IProblem, double, ZDT1, zdt1)

}  // namespace Eacpp