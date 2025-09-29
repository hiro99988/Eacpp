#pragma once

#include <Eigen/Core>

#include "Problems/ZDTBase.h"
#include "Reflections/Reflection.h"

namespace eacpp {

class ZDT2 : public ZDTBase {
   public:
    ZDT2() : ZDTBase() {}
    ZDT2(int decisionVariablesNum) : ZDTBase(decisionVariablesNum) {}

   private:
    double F1(double x1) const override;
    double G(const Eigen::ArrayXd& x) const override;
    double H(double f1, double g) const override;
};

REGISTER_REFLECTION(IBenchmark, ZDT2, zdt2)
REGISTER_TEMPLATE_REFLECTION(IProblem, double, ZDT2, zdt2)

}  // namespace eacpp