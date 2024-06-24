#ifndef UniformRandomSampling_h
#define UniformRandomSampling_h

#include <eigen3/Eigen/Core>

#include "Samplings/SamplingBase.h"
#include "Utils/TemplateType.h"

namespace Eacpp {

class UniformRandomSampling : public SamplingBase<double> {
   public:
    double min = 0.0;
    double max = 1.0;

    UniformRandomSampling() : SamplingBase() {}
    UniformRandomSampling(double min, double max) : SamplingBase(), min(min), max(max) {
        if (min > max) {
            throw std::invalid_argument("min must be less than or equal to max");
        };
    }
    UniformRandomSampling(double min, double max, SeedType seed) : SamplingBase(seed), min(min), max(max) {
        if (min > max) {
            throw std::invalid_argument("min must be less than or equal to max");
        };
    }

    Eigen::ArrayXXd Sample(int sampleNum, int variableNum) const override;
};

}  // namespace Eacpp

#endif