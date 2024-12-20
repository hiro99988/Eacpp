#pragma once

#include <eigen3/Eigen/Core>
#include <memory>
#include <tuple>
#include <vector>

#include "Individual.h"
#include "Rng/IRng.h"
#include "Samplings/RandomSamplingBase.h"
#include "Utils/Utils.h"

namespace Eacpp {

class RealRandomSampling : public RandomSamplingBase<double> {
   public:
    RealRandomSampling(double min, double max) {
        swapIfMaxLessThanMin(min, max);
        variableBounds = {{min, max}};
    }
    RealRandomSampling(double min, double max, const std::shared_ptr<IRng>& rng)
        : RandomSamplingBase(rng) {
        swapIfMaxLessThanMin(min, max);
        variableBounds = {{min, max}};
    }
    RealRandomSampling(const std::pair<double, double>& variableBound)
        : variableBounds({variableBound}) {}
    RealRandomSampling(const std::pair<double, double>& variableBound,
                       const std::shared_ptr<IRng>& rng)
        : RandomSamplingBase(rng), variableBounds({variableBound}) {}
    RealRandomSampling(
        const std::vector<std::pair<double, double>>& variableBounds)
        : variableBounds(variableBounds) {}
    RealRandomSampling(
        const std::vector<std::pair<double, double>>& variableBounds,
        const std::shared_ptr<IRng>& rng)
        : RandomSamplingBase(rng), variableBounds(variableBounds) {}

    const std::vector<std::pair<double, double>>& VariableBounds() const {
        return variableBounds;
    }

    std::vector<Individuald> Sample(const int sampleNum,
                                    const int variableNum) const override;

   private:
    std::vector<std::pair<double, double>> variableBounds;
};

}  // namespace Eacpp
