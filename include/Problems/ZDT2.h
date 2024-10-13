#pragma once

#include <eigen3/Eigen/Core>

#include "Problems/ZDTBase.h"
#include "Reflections/Reflection.h"

namespace Eacpp {

class ZDT2 : public ZDTBase {
   public:
    ZDT2() : ZDTBase() {}

   private:
    double F1(double x1) const override;
    double G(const Eigen::ArrayXd& x) const override;
    double F2(double f1, double g) const override;

#ifdef _TEST_
   public:
    friend class ZDT2Test;
#endif
};

REGISTER_REFLECTION(IBenchmark, ZDT2)

}  // namespace Eacpp