#pragma once

#include <Eigen/Core>
#include <vector>

#include "Problems/ZDTBase.h"
#include "Reflections/Reflection.h"

namespace eacpp {

class ZDT6 : public ZDTBase {
   public:
    ZDT6() : ZDTBase(10) {}
    ZDT6(int decisionVariablesNum) : ZDTBase(decisionVariablesNum) {}

   private:
    double F1(double x1) const override;
    double G(const Eigen::ArrayXd& x) const override;
    double H(double f1, double g) const override;
    std::vector<Eigen::ArrayXd> GenerateParetoFront(
        int pointsNum) const override;
};

REGISTER_REFLECTION(IBenchmark, ZDT6, zdt6)
REGISTER_TEMPLATE_REFLECTION(IProblem, double, ZDT6, zdt6)

}  // namespace eacpp