#ifndef RandomSamplingBase_h
#define RandomSamplingBase_h

#include "Rng/Rng.h"
#include "Samplings/ISampling.h"
#include "Utils/TemplateType.h"

namespace Eacpp {

template <Number T>
class RandomSamplingBase : public ISampling<T> {
   public:
    RandomSamplingBase() : _rng() {}
    RandomSamplingBase(SeedType seed) : _rng(seed) {}
    virtual ~RandomSamplingBase() {}

   protected:
    Rng _rng;
};

}  // namespace Eacpp

#endif