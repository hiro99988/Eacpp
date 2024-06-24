#ifndef SamplingBase_h
#define SamplingBase_h

#include "Samplings/ISampling.h"
#include "Utils/Rng.h"
#include "Utils/TemplateType.h"

namespace Eacpp {

template <Number T>
class SamplingBase : public ISampling<T> {
   public:
    SamplingBase() : _rng() {}
    SamplingBase(SeedType seed) : _rng(seed) {}

   protected:
    Rng _rng;
};

}  // namespace Eacpp

#endif