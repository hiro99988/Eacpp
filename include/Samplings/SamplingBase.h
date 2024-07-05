#ifndef SamplingBase_h
#define SamplingBase_h

#include "Rng/Rng.h"
#include "Samplings/ISampling.h"
#include "Utils/TemplateType.h"

namespace Eacpp {

template <Number T>
class SamplingBase : public ISampling<T> {
   public:
    SamplingBase() : _rng() {}
    SamplingBase(SeedType seed) : _rng(seed) {}
    virtual ~SamplingBase() {}

   protected:
    Rng _rng;
};

}  // namespace Eacpp

#endif