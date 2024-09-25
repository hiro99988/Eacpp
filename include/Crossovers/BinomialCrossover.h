#pragma once

#include <eigen3/Eigen/Core>
#include <memory>
#include <vector>

#include "Crossovers/CrossoverBase.h"
#include "Individual/Individual.h"
#include "Rng/IRng.h"
namespace Eacpp {

class BinomialCrossover : public CrossoverBase<double> {
   public:
    double scalingFactor;

    BinomialCrossover(double crossoverRate, double scalingFactor)
        : CrossoverBase(3, crossoverRate), scalingFactor(scalingFactor) {}
    BinomialCrossover(double crossoverRate, double scalingFactor, std::shared_ptr<IRng> rng)
        : CrossoverBase(3, crossoverRate, rng), scalingFactor(scalingFactor) {}

   private:
    Individuald performCrossover(const std::vector<Individuald>& parents) const override;
};

}  // namespace Eacpp
