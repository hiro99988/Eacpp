#ifndef CrossoverBase_H
#define CrossoverBase_H

#include <eigen3/Eigen/Core>
#include <memory>
#include <vector>

#include "Crossovers/ICrossover.h"
#include "Rng/IRng.h"
#include "Rng/Rng.h"
#include "Utils/TemplateType.h"

namespace Eacpp {

template <Number T>
class CrossoverBase : public ICrossover<T> {
   public:
    double crossoverRate;

    CrossoverBase(int parentNum, double crossoverRate) : _parentNum(parentNum), crossoverRate(crossoverRate) {
        _rng = std::make_shared<Rng>();
    }
    CrossoverBase(int parentNum, double crossoverRate, std::shared_ptr<IRng> rng)
        : _parentNum(parentNum), crossoverRate(crossoverRate), _rng(rng) {}
    virtual ~CrossoverBase() {}

    int GetParentNum() const override { return _parentNum; }

    Eigen::ArrayX<T> Cross(const std::vector<Eigen::ArrayX<T>>& parents) const override {
        int actualParentNum = parents.size();
        if (actualParentNum != _parentNum) {
            throw std::invalid_argument("Invalid size of parents");
        }
        return performCrossover(parents);
    }

   protected:
    std::shared_ptr<IRng> _rng;

    virtual Eigen::ArrayX<T> performCrossover(const std::vector<Eigen::ArrayX<T>>& parents) const = 0;

   private:
    int _parentNum;
};

}  // namespace Eacpp

#endif