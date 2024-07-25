#ifndef _POLYNOMIAL_MUTATION_H_
#define _POLYNOMIAL_MUTATION_H_

#include <array>
#include <eigen3/Eigen/Core>
#include <memory>
#include <vector>

#include "Mutations/MutationBase.h"

namespace Eacpp {

class PolynomialMutation : public MutationBase<double> {
   public:
    double distributionIndex;
    std::vector<std::array<double, 2>> variableBounds;

    PolynomialMutation(double mutationRate, double distributionIndex, std::vector<std::array<double, 2>> variableBounds)
        : MutationBase(mutationRate),
          distributionIndex(distributionIndex),
          variableBounds(variableBounds),
          _length(variableBounds.size()),
          _lastBoundIndex(variableBounds.size() - 1) {}
    PolynomialMutation(double mutationRate, double distributionIndex, std::vector<std::array<double, 2>> variableBounds,
                       std::shared_ptr<IRng> rng)
        : MutationBase(mutationRate, rng),
          distributionIndex(distributionIndex),
          variableBounds(variableBounds),
          _length(variableBounds.size()),
          _lastBoundIndex(variableBounds.size() - 1) {}

    void Mutate(Eigen::ArrayXd& individual) const override;

   private:
    int _length;
    int _lastBoundIndex;

    double Sigma() const;
};

}  // namespace Eacpp

#endif