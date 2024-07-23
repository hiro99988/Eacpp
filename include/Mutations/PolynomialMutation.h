#ifndef _POLYNOMIAL_MUTATION_H_
#define _POLYNOMIAL_MUTATION_H_

#include <array>
#include <eigen3/Eigen/Core>
#include <vector>

#include "Mutations/IMutation.h"

namespace Eacpp {

class PolynomialMutation : public IMutation<double> {
   public:
    double mutationRate;
    double distributionIndex;
    std::vector<std::array<double, 2>> variableBounds;

    PolynomialMutation(double mutationRate, double distributionIndex, std::vector<std::array<double, 2>> variableBounds)
        : mutationRate(mutationRate),
          distributionIndex(distributionIndex),
          variableBounds(variableBounds),
          _length(variableBounds.size()),
          _lastBoundIndex(variableBounds.size() - 1) {}
    PolynomialMutation(double mutationRate, double distributionIndex, std::vector<std::array<double, 2>> variableBounds,
                       IRng* rng)
        : IMutation(rng),
          mutationRate(mutationRate),
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