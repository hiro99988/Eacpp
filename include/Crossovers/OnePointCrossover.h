#ifndef OnePointCrossover_H
#define OnePointCrossover_H

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

   private:
    Rng _rng;

    std::vector<Eigen::VectorX<T>> performCrossover(std::vector<Eigen::VectorX<T>> parents) override;
};

}  // namespace Eacpp

#endif