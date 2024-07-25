#ifndef _RANDOM_SAMPLING_BASE_H
#define _RANDOM_SAMPLING_BASE_H

#include <memory>

#include "Rng/IRng.h"
#include "Rng/Rng.h"
#include "Samplings/ISampling.h"
#include "Utils/TemplateType.h"

namespace Eacpp {

template <Number T>
class RandomSamplingBase : public ISampling<T> {
   public:
    RandomSamplingBase() { _rng = std::make_shared<Rng>(); }
    explicit RandomSamplingBase(std::shared_ptr<IRng> rng) : _rng(rng) {}
    virtual ~RandomSamplingBase() {}

   protected:
    std::shared_ptr<IRng> _rng;
};

}  // namespace Eacpp

#endif