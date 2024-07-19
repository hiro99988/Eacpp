#ifndef RandomSelection_h
#define RandomSelection_h

#include <eigen3/Eigen/Core>

#include "Rng/HasRng.h"
#include "Rng/IRng.h"
#include "Selections/ISelection.h"
#include "Utils/TemplateType.h"

namespace Eacpp {

template <Number T>
class RandomSelection : public ISelection<T>, protected HasRng {
   public:
    RandomSelection() {}
    RandomSelection(IRng* rng) : HasRng(rng) {}

    Eigen::ArrayXX<T> Select(int parentNum, const Eigen::ArrayXX<T>& population) const override {
        return _rng->Choice(population, parentNum, false);
    };
};

}  // namespace Eacpp

#endif