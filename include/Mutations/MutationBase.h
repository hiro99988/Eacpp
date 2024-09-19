#pragma once

#include <memory>

#include "Mutations/IMutation.h"
#include "Rng/IRng.h"
#include "Rng/Rng.h"

namespace Eacpp {

template <typename T>
class MutationBase : public IMutation<T> {
   public:
    double mutationRate;

    explicit MutationBase(double mutationRate) : mutationRate(mutationRate) { _rng = std::make_shared<Rng>(); }
    MutationBase(double mutationRate, std::shared_ptr<IRng> rng) : mutationRate(mutationRate), _rng(rng) {}
    virtual ~MutationBase() {}

   protected:
    std::shared_ptr<IRng> _rng;
};

}  // namespace Eacpp
