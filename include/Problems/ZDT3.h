#pragma once

#include <eigen3/Eigen/Core>
#include <vector>

#include "Problems/ZDTBase.h"
#include "Reflections/Reflection.h"

namespace Eacpp {

class ZDT3 : public ZDTBase {
   public:
    ZDT3() : ZDTBase() {}
    ZDT3(int decisionVariablesNum) : ZDTBase(decisionVariablesNum) {}

   private:
    double F1(double x1) const override;
    double G(const Eigen::ArrayXd& x) const override;
    double H(double f1, double g) const override;
    std::vector<Eigen::ArrayXd> GenerateParetoFront(
        int pointsNum) const override;

#ifdef _TEST_
   public:
    friend class ZDT3Test;
#endif
};

REGISTER_REFLECTION(IBenchmark, ZDT3, zdt3)
REGISTER_TEMPLATE_REFLECTION(IProblem, double, ZDT3, zdt3)

}  // namespace Eacpp