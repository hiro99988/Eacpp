#ifndef _RANDOM_SELECTION_H
#define _RANDOM_SELECTION_H

#include <eigen3/Eigen/Core>
#include <memory>

#include "Rng/IRng.h"
#include "Rng/Rng.h"
#include "Selections/ISelection.h"
#include "Utils/TemplateType.h"

namespace Eacpp {

template <Number T>
class RandomSelection : public ISelection<T> {
   public:
    RandomSelection() {}
    explicit RandomSelection(std::shared_ptr<IRng> rng) : _rng(rng) {}

    Eigen::ArrayX<T> Select(int parentNum, const Eigen::ArrayX<T>& population) const override {
        return _rng->Choice(population, parentNum, false);
    };

   private:
    std::shared_ptr<IRng> _rng;
};

}  // namespace Eacpp

#endif