#ifndef UniformRandomSampling_h
#define UniformRandomSampling_h

#include <eigen3/Eigen/Core>
#include <memory>
#include <vector>

#include "Rng/IRng.h"
#include "Samplings/RandomSamplingBase.h"
#include "Utils/TemplateType.h"

namespace Eacpp {

class UniformRandomSampling : public RandomSamplingBase<double> {
   public:
    double min = 0.0;
    double max = 1.0;

    UniformRandomSampling() : RandomSamplingBase() {}
    UniformRandomSampling(double min, double max) : min(min), max(max) { CheckMinMax(min, max); }
    UniformRandomSampling(double min, double max, std::shared_ptr<IRng> rng) : RandomSamplingBase(rng), min(min), max(max) {
        CheckMinMax(min, max);
    }

    std::vector<Eigen::ArrayXd> Sample(const int sampleNum, const int variableNum) const override;

   private:
    void CheckMinMax(const int min, const int max) const {
        if (min > max) {
            throw std::invalid_argument("min must be less than or equal to max");
        }
    }
};

}  // namespace Eacpp

#endif