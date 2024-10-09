#pragma once

#include <cstdint>
#include <eigen3/Eigen/Core>
#include <memory>
#include <vector>

#include "Crossovers/CrossoverBase.h"
#include "Individual/Individual.h"
#include "Rng/IRng.h"

namespace Eacpp {

class SimulatedBinaryCrossover : public CrossoverBase<double> {
   public:
    explicit SimulatedBinaryCrossover(double crossoverRate) : CrossoverBase<double>(2, crossoverRate) {}
    SimulatedBinaryCrossover(double crossoverRate, std::shared_ptr<IRng> rng) : CrossoverBase<double>(2, crossoverRate, rng) {}
};

}  // namespace Eacpp
