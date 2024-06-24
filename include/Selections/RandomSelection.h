#ifndef RandomSelection_h
#define RandomSelection_h

#include "Selections/ISelection.h"
#include "Utils/Rng.h"
#include "Utils/TemplateType.h"

namespace Eacpp {

template <Number T>
class RandomSelection : public ISelection<T> {
   public:
    RandomSelection() : _rng() {}
    RandomSelection(SeedType seed) : _rng(seed) {}

    Eigen::ArrayXX<T> select(int parentNum, const Eigen::ArrayXX<T>& population) const override;

   private:
    Rng _rng;
};

}  // namespace Eacpp

#endif