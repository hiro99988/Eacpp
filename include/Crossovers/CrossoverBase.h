#ifndef CrossoverBase_H
#define CrossoverBase_H

#include <eigen3/Eigen/Core>

#include "Crossovers/ICrossover.h"
#include "Rng/HasRng.h"
#include "Rng/IRng.h"
#include "Utils/TemplateType.h"

namespace Eacpp {

template <Number T>
class CrossoverBase : public ICrossover<T>, protected HasRng {
   public:
    double crossoverRate;

    CrossoverBase(int parentNum, double crossoverRate) : _parentNum(parentNum), crossoverRate(crossoverRate) {}
    CrossoverBase(int parentNum, double crossoverRate, IRng* rng)
        : _parentNum(parentNum), crossoverRate(crossoverRate), HasRng(rng) {}
    virtual ~CrossoverBase() {}

    int GetParentNum() const override { return _parentNum; }

    Eigen::ArrayX<T> Cross(const Eigen::ArrayXX<T>& parents) const override {
        int actualParentNum = parents.cols();
        if (actualParentNum != _parentNum) {
            throw std::invalid_argument("Invalid size of parents");
        }
        return performCrossover(parents);
    }

   protected:
    virtual Eigen::ArrayX<T> performCrossover(const Eigen::ArrayXX<T>& parents) const = 0;

   private:
    int _parentNum;
};

}  // namespace Eacpp

#endif