#pragma once

#include <eigen3/Eigen/Core>

#include "Problems/BenchmarkReflection.h"
#include "Problems/ZDTBase.h"

namespace Eacpp {

class ZDT1 : public ZDTBase {
   public:
    ZDT1() : ZDTBase(30) {}

   private:
    double F1(double x1) const override;
    double G(const Eigen::ArrayXd& x) const override;
    double F2(double f1, double g) const override;

#ifdef _TEST_
   public:
    friend class ZDT1Test;
#endif
};

REGISTER_BENCHMARK_REFLECTION(ZDT1)

}  // namespace Eacpp