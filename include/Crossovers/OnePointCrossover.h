#ifndef OnePointCrossover_H
#define OnePointCrossover_H

#include <cstdint>
#include <eigen3/Eigen/Core>
#include <memory>
#include <vector>

#include "Crossovers/CrossoverBase.h"
#include "Rng/IRng.h"
#include "Rng/Rng.h"

namespace Eacpp {

template <typename T>
class OnePointCrossover : public CrossoverBase<T> {
   public:
    explicit OnePointCrossover(double crossoverRate) : CrossoverBase<T>(2, crossoverRate) {}
    OnePointCrossover(double crossoverRate, std::shared_ptr<IRng> rng) : CrossoverBase<T>(2, crossoverRate, rng) {}

   private:
    Eigen::ArrayX<T> performCrossover(const std::vector<Eigen::ArrayX<T>>& parents) const override {
        if (this->_rng->Random() > this->crossoverRate) {
            return parents[0];
        }
        int size = parents[0].size();
        int crossoverPoint = this->_rng->Integer(1, size - 1);
        Eigen::ArrayX<T> child(size);
        child << parents[0].head(crossoverPoint), parents[1].tail(size - crossoverPoint);
        return child;
    }
};

}  // namespace Eacpp

#endif