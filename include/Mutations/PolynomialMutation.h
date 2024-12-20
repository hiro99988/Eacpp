#pragma once

#include <eigen3/Eigen/Core>
#include <memory>
#include <tuple>
#include <vector>

#include "Individual.h"
#include "Mutations/MutationBase.h"

namespace Eacpp {

class PolynomialMutation : public MutationBase<double> {
   public:
    PolynomialMutation(double mutationRate,
                       std::vector<std::pair<double, double>> variableBounds)
        : MutationBase(mutationRate),
          distributionIndex(DefaultDistributionIndex),
          variableBounds(variableBounds) {}
    PolynomialMutation(double mutationRate, double distributionIndex,
                       std::vector<std::pair<double, double>> variableBounds)
        : MutationBase(mutationRate),
          distributionIndex(distributionIndex),
          variableBounds(variableBounds) {}
    PolynomialMutation(double mutationRate,
                       std::vector<std::pair<double, double>> variableBounds,
                       std::shared_ptr<IRng> rng)
        : MutationBase(mutationRate, rng),
          distributionIndex(DefaultDistributionIndex),
          variableBounds(variableBounds) {}
    PolynomialMutation(double mutationRate, double distributionIndex,
                       std::vector<std::pair<double, double>> variableBounds,
                       std::shared_ptr<IRng> rng)
        : MutationBase(mutationRate, rng),
          distributionIndex(distributionIndex),
          variableBounds(variableBounds) {}

    void Mutate(Individuald& individual) const override;

   private:
    constexpr static double DefaultDistributionIndex = 20.0;

    double distributionIndex;
    std::vector<std::pair<double, double>> variableBounds;

    double PerformMutation(double y, double lower, double upper) const;

#ifdef _TEST_
   public:
    friend class PolynomialMutationTest;
#endif
};

}  // namespace Eacpp
