#ifndef OnePointCrossover_H
#define OnePointCrossover_H

#include <cstdint>
#include <eigen3/Eigen/Core>
#include <vector>

#include "Crossovers/CrossoverBase.h"
#include "Rng/IRng.h"
#include "Rng/Rng.h"
#include "Utils/TemplateType.h"

namespace Eacpp {

template <Number T>
class OnePointCrossover : public CrossoverBase<T> {
   public:
    double crossoverRate;

    OnePointCrossover(double crossoverRate) : CrossoverBase<T>(2), crossoverRate(crossoverRate) {
        _rng = new Rng();
        _isRngCreated = true;
    }
    explicit OnePointCrossover(double crossoverRate, IRng* rng)
        : CrossoverBase<T>(2), crossoverRate(crossoverRate), _rng(rng) {}
    ~OnePointCrossover() {
        if (_isRngCreated) {
            delete _rng;
        }
    }

   private:
    IRng* _rng;
    bool _isRngCreated = false;

    Eigen::ArrayX<T> performCrossover(const Eigen::ArrayXX<T>& parents) const override {
        if (_rng->Random() > crossoverRate) {
            return parents.col(0);
        }
        int size = parents.rows();
        int crossoverPoint = _rng->Integer(1, size - 1);
        Eigen::ArrayX<T> child(size);
        child << parents.block(0, 0, crossoverPoint, 1), parents.block(crossoverPoint, 1, size - crossoverPoint, 1);
        return child;
    }
};

}  // namespace Eacpp

#endif