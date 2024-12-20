#pragma once

#include <eigen3/Eigen/Core>
#include <memory>

#include "Individual.h"
#include "Mutations/MutationBase.h"
#include "Rng/IRng.h"

namespace Eacpp {
class BitFlipMutation : public MutationBase<int> {
   public:
    explicit BitFlipMutation(double mutationRate)
        : MutationBase(mutationRate) {}
    BitFlipMutation(double mutationRate, std::shared_ptr<IRng> rng)
        : MutationBase(mutationRate, rng) {}

    void Mutate(Individuali& individual) const override;
};

}  // namespace Eacpp
