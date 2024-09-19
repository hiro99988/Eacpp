#pragma once

#include <eigen3/Eigen/Core>
#include <memory>
#include <vector>

#include "Crossovers/CrossoverBase.h"
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
    Eigen::ArrayXd performCrossover(const std::vector<Eigen::ArrayXd>& parents) const override;
};

}  // namespace Eacpp
