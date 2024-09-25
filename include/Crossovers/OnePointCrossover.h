#pragma once

#include <cstdint>
#include <eigen3/Eigen/Core>
#include <memory>
#include <vector>

#include "Crossovers/CrossoverBase.h"
#include "Individual/Individual.h"
#include "Rng/IRng.h"

namespace Eacpp {

template <typename T>
class OnePointCrossover : public CrossoverBase<T> {
   public:
    explicit OnePointCrossover(double crossoverRate) : CrossoverBase<T>(2, crossoverRate) {}
    OnePointCrossover(double crossoverRate, std::shared_ptr<IRng> rng) : CrossoverBase<T>(2, crossoverRate, rng) {}

   private:
    Individual<T> performCrossover(const std::vector<Individual<T>>& parents) const override {
        if (this->_rng->Random() > this->crossoverRate) {
            return Individual<T>(parents[0].solution);
        }
        int size = parents[0].solution.size();
        int crossoverPoint = this->_rng->Integer(1, size - 1);
        Individual<T> child(size);
        child.solution << parents[0].solution.head(crossoverPoint), parents[1].solution.tail(size - crossoverPoint);
        return child;
    }
};

}  // namespace Eacpp
