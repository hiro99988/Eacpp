#pragma once

#include <eigen3/Eigen/Core>
#include <memory>
#include <tuple>
#include <vector>

#include "Crossovers/CrossoverBase.h"
#include "Individual.h"
#include "Rng/IRng.h"

namespace Eacpp {

class SimulatedBinaryCrossover : public CrossoverBase<double> {
   public:
    static constexpr int ParentNum = 2;
    static constexpr double DefaultDistributionIndex = 20.0;

    SimulatedBinaryCrossover(double crossoverRate, const std::vector<std::pair<double, double>>& variableBounds)
        : CrossoverBase<double>(ParentNum, crossoverRate),
          distributionIndex(DefaultDistributionIndex),
          variableBounds(variableBounds) {}
    SimulatedBinaryCrossover(double crossoverRate, double distributionIndex,
                             const std::vector<std::pair<double, double>>& variableBounds)
        : CrossoverBase<double>(ParentNum, crossoverRate),
          distributionIndex(distributionIndex),
          variableBounds(variableBounds) {}
    SimulatedBinaryCrossover(double crossoverRate, const std::vector<std::pair<double, double>>& variableBounds,
                             const std::shared_ptr<IRng>& rng)
        : CrossoverBase<double>(ParentNum, crossoverRate, rng),
          distributionIndex(DefaultDistributionIndex),
          variableBounds(variableBounds) {}
    SimulatedBinaryCrossover(double crossoverRate, double distributionIndex,
                             const std::vector<std::pair<double, double>>& variableBounds, const std::shared_ptr<IRng>& rng)
        : CrossoverBase<double>(ParentNum, crossoverRate, rng),
          distributionIndex(distributionIndex),
          variableBounds(variableBounds) {}

   private:
    double distributionIndex;
    std::vector<std::pair<double, double>> variableBounds;

    Individuald performCrossover(const std::vector<Individuald>& parents) const override;
    double Betaq(double alpha) const;
    double Alpha(double beta) const;
    double Beta1(double x1, double x2, double lowerBound) const;
    double Beta2(double x1, double x2, double upperBound) const;

#ifdef _TEST_
   public:
    friend class SimulatedBinaryCrossoverTest;
#endif
};

}  // namespace Eacpp
