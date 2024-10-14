#pragma once

#include <memory>

#include "Individual/Individual.h"
#include "Rng/IRng.h"
#include "Rng/Rng.h"
#include "Samplings/ISampling.h"

namespace Eacpp {

template <typename T>
class RandomSamplingBase : public ISampling<T> {
   public:
    RandomSamplingBase() {
        _rng = std::make_shared<Rng>();
    }
    explicit RandomSamplingBase(const std::shared_ptr<IRng>& rng) : _rng(rng) {}
    virtual ~RandomSamplingBase() {}

   protected:
    std::shared_ptr<IRng> _rng;
};

}  // namespace Eacpp
