#ifndef BitFlipMutation_h
#define BitFlipMutation_h

#include <eigen3/Eigen/Core>
#include <iostream>

#include "Mutations/IMutation.h"
#include "Rng/IRng.h"

namespace Eacpp {
class BitFlipMutation : public IMutation<int> {
   public:
    double mutationRate;

    explicit BitFlipMutation(double mutationRate) : mutationRate(mutationRate) {}
    BitFlipMutation(double mutationRate, IRng* rng) : mutationRate(mutationRate), IMutation(rng) {}

    void Mutate(Eigen::ArrayXi& individual) const override;
};

}  // namespace Eacpp

#endif