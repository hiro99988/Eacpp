#ifndef _RANDOM_SELECTION_H
#define _RANDOM_SELECTION_H

#include <eigen3/Eigen/Core>
#include <memory>

#include "Rng/IRng.h"
#include "Rng/Rng.h"
#include "Selections/ISelection.h"
#include "Utils/TemplateType.h"

namespace Eacpp {

class RandomSelection : public ISelection {
   public:
    RandomSelection() { _rng = std::make_shared<Rng>(); }
    explicit RandomSelection(std::shared_ptr<IRng> rng) : _rng(rng) {}

    std::vector<int> Select(int parentNum, const std::vector<int>& population) const override {
        return _rng->Choice(population, parentNum, false);
    };

   private:
    std::shared_ptr<IRng> _rng;
};

}  // namespace Eacpp

#endif