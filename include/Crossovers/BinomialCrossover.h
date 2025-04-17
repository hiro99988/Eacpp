#pragma once

#include <Eigen/Core>
#include <memory>
#include <vector>

#include "Crossovers/CrossoverBase.h"
#include "Individual.h"
#include "Rng/IRng.h"
namespace Eacpp {

class BinomialCrossover : public CrossoverBase<double> {
   public:
    static constexpr int ParentNum = 3;
    static constexpr double DefaultScalingFactor = 0.5;

    double scalingFactor;

    explicit BinomialCrossover(double crossoverRate)
        : CrossoverBase(ParentNum, crossoverRate),
          scalingFactor(DefaultScalingFactor) {}
    BinomialCrossover(double crossoverRate, double scalingFactor)
        : CrossoverBase(ParentNum, crossoverRate),
          scalingFactor(scalingFactor) {}
    BinomialCrossover(double crossoverRate, std::shared_ptr<IRng> rng)
        : CrossoverBase(ParentNum, crossoverRate, rng),
          scalingFactor(DefaultScalingFactor) {}
    BinomialCrossover(double crossoverRate, double scalingFactor,
                      std::shared_ptr<IRng> rng)
        : CrossoverBase(ParentNum, crossoverRate, rng),
          scalingFactor(scalingFactor) {}

   private:
    Individuald performCrossover(
        const std::vector<Individuald>& parents) const override;
};

}  // namespace Eacpp
