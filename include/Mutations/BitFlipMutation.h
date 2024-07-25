#ifndef BitFlipMutation_h
#define BitFlipMutation_h

#include <eigen3/Eigen/Core>
#include <memory>

#include "Mutations/MutationBase.h"
#include "Rng/IRng.h"

namespace Eacpp {
class BitFlipMutation : public MutationBase<int> {
   public:
    explicit BitFlipMutation(double mutationRate) : MutationBase(mutationRate) {}
    BitFlipMutation(double mutationRate, std::shared_ptr<IRng> rng) : MutationBase(mutationRate, rng) {}

    void Mutate(Eigen::ArrayXi& individual) const override;
};

}  // namespace Eacpp

#endif