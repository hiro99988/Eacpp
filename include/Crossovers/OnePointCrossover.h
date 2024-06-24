#ifndef OnePointCrossover_H
#define OnePointCrossover_H

#include <cstdint>
#include <eigen3/Eigen/Core>
#include <vector>

#include "Crossovers/CrossoverBase.h"
#include "Utils/Rng.h"
#include "Utils/TemplateType.h"

namespace Eacpp {

template <Number T>
class OnePointCrossover : public CrossoverBase<T> {
   public:
    OnePointCrossover() : CrossoverBase<T>(2, 1), _rng() {}
    explicit OnePointCrossover(SeedType seed) : CrossoverBase<T>(2, 1), _rng(seed) {}

   private:
    Rng _rng;

    Eigen::ArrayXX<T> performCrossover(const Eigen::ArrayXX<T>& parents) const override;
};

}  // namespace Eacpp

#endif