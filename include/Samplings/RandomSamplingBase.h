#ifndef RandomSamplingBase_h
#define RandomSamplingBase_h

#include "Rng/HasRng.h"
#include "Rng/IRng.h"
#include "Samplings/ISampling.h"
#include "Utils/TemplateType.h"

namespace Eacpp {

template <Number T>
class RandomSamplingBase : public ISampling<T>, protected HasRng {
   public:
    RandomSamplingBase() {}
    explicit RandomSamplingBase(IRng* rng) : HasRng(rng) {}
    virtual ~RandomSamplingBase() {}
};

}  // namespace Eacpp

#endif