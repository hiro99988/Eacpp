#pragma once

#include <eigen3/Eigen/Core>
#include <tuple>
#include <vector>

#include "Individual.h"
#include "Problems/ZDTBase.h"
#include "Reflections/Reflection.h"

namespace Eacpp {

class ZDT4 : public ZDTBase {
   public:
    ZDT4() : ZDTBase(10, {{0.0, 1.0}, {-5.0, 5.0}}) {}
    ZDT4(int decisionVariablesNum)
        : ZDTBase(decisionVariablesNum, {{0.0, 1.0}, {-5.0, 5.0}}) {}

   private:
    double F1(double x1) const override;
    double G(const Eigen::ArrayXd& x) const override;
    double H(double f1, double g) const override;
    std::vector<bool> EvaluateConstraints(
        const Individuald& individual) const override;

#ifdef _TEST_
   public:
    friend class ZDT4Test;
#endif
};

REGISTER_REFLECTION(IBenchmark, ZDT4, zdt4)
REGISTER_TEMPLATE_REFLECTION(IProblem, double, ZDT4, zdt4)

}  // namespace Eacpp