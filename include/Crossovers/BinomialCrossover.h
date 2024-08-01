#ifndef _BINOMIAL_CROSSOVER_H_
#define _BINOMIAL_CROSSOVER_H_

#include <eigen3/Eigen/Core>
#include <memory>
#include <vector>

#include "Crossovers/CrossoverBase.h"
#include "Rng/IRng.h"
#include "Rng/Rng.h"
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

#endif
