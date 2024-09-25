#pragma once

#include <eigen3/Eigen/Core>
#include <memory>
#include <tuple>
#include <vector>

#include "Individual/Individual.h"
#include "Mutations/MutationBase.h"

namespace Eacpp {

class PolynomialMutation : public MutationBase<double> {
   public:
    double distributionIndex;
    std::vector<std::pair<double, double>> variableBounds;

    PolynomialMutation(double mutationRate, double distributionIndex, std::vector<std::pair<double, double>> variableBounds)
        : MutationBase(mutationRate),
          distributionIndex(distributionIndex),
          variableBounds(variableBounds),
          _lastBoundIndex(variableBounds.size() - 1) {}

    PolynomialMutation(double mutationRate, double distributionIndex, std::vector<std::pair<double, double>> variableBounds,
                       std::shared_ptr<IRng> rng)
        : MutationBase(mutationRate, rng),
          distributionIndex(distributionIndex),
          variableBounds(variableBounds),
          _lastBoundIndex(variableBounds.size() - 1) {}

    void Mutate(Individuald& individual) const override;

   private:
    int _lastBoundIndex;

    void PerformMutation(int index, Individuald& individual, double sigma) const;
    double Sigma() const;

#ifdef _TEST_
   public:
    friend class PolynomialMutationTest;
#endif
};

}  // namespace Eacpp
