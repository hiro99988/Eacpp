#pragma once

#include <eigen3/Eigen/Core>
#include <memory>
#include <stdexcept>
#include <vector>

#include "Crossovers/ICrossover.h"
#include "Individual.h"
#include "Rng/IRng.h"
#include "Rng/Rng.h"

namespace Eacpp {

template <typename T>
class CrossoverBase : public ICrossover<T> {
   public:
    double crossoverRate;

    CrossoverBase(int parentNum, double crossoverRate) : _parentsNum(parentNum), crossoverRate(crossoverRate) {
        _rng = std::make_shared<Rng>();
    }
    CrossoverBase(int parentNum, double crossoverRate, std::shared_ptr<IRng> rng)
        : _parentsNum(parentNum), crossoverRate(crossoverRate), _rng(rng) {}
    virtual ~CrossoverBase() {}

    int GetParentNum() const override {
        return _parentsNum;
    }

    Individual<T> Cross(const std::vector<Individual<T>>& parents) const override {
        int actualParentNum = parents.size();
        if (actualParentNum != _parentsNum) {
            throw std::invalid_argument("Invalid size of parents");
        }
        return performCrossover(parents);
    }

   protected:
    std::shared_ptr<IRng> _rng;

    virtual Individual<T> performCrossover(const std::vector<Individual<T>>& parents) const = 0;

   private:
    int _parentsNum;
};

}  // namespace Eacpp
