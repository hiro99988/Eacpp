#pragma once

#include <eigen3/Eigen/Core>
#include <memory>
#include <vector>

#include "Crossovers/CrossoverBase.h"
#include "Individual/Individual.h"
#include "Rng/IRng.h"

namespace Eacpp {

class SimulatedBinaryCrossover : public CrossoverBase<double> {
   public:
    static constexpr int ParentNum = 2;
    static constexpr double DefaultDistributionIndex = 20.0;

    double distributionIndex;

    explicit SimulatedBinaryCrossover(double crossoverRate)
        : CrossoverBase<double>(ParentNum, crossoverRate), distributionIndex(DefaultDistributionIndex) {}
    SimulatedBinaryCrossover(double crossoverRate, double distributionIndex)
        : CrossoverBase<double>(ParentNum, crossoverRate), distributionIndex(distributionIndex) {}
    SimulatedBinaryCrossover(double crossoverRate, std::shared_ptr<IRng> rng)
        : CrossoverBase<double>(ParentNum, crossoverRate, rng), distributionIndex(DefaultDistributionIndex) {}
    SimulatedBinaryCrossover(double crossoverRate, double distributionIndex, std::shared_ptr<IRng> rng)
        : CrossoverBase<double>(ParentNum, crossoverRate, rng), distributionIndex(distributionIndex) {}

   private:
    Individuald performCrossover(const std::vector<Individuald>& parents) const override;
    double Beta() const;

#ifdef _TEST_
   public:
    friend class SimulatedBinaryCrossoverTest;
#endif
};

}  // namespace Eacpp
