#ifndef BinomialCrossover_H
#define BinomialCrossover_H

#include "Crossovers/CrossoverBase.h"
#include "Rng/IRng.h"
#include "Rng/Rng.h"
#include "Utils/TemplateType.h"

namespace Eacpp {

class BinomialCrossover : public CrossoverBase<double> {
   public:
    double F;

    explicit BinomialCrossover(double crossoverRate, double F) : CrossoverBase(3, crossoverRate), F(F) {}
    virtual ~BinomialCrossover() {}

   protected:
    Eigen::ArrayX<double> performCrossover(const Eigen::ArrayXX<double>& parents) const override {
        Eigen::ArrayX<double> child = parents.col(0);
        int jr = _rng.integers(child.size());
        for (int i = 0; i < child.size(); i++) {
            if (_rng.random() < crossoverRate || i == jr) {
                child[i] += F * (parents(1, i) - parents(2, i));
            }
        }
        return child;
    }
};

}  // namespace Eacpp

#endif

// import numpy as np

// from src.crossovers.crossover_base import CrossoverBase

// class BinomialCrossover(CrossoverBase):
//     def __init__(self, crossover_rate: float, F: float, seed: int = None):
//         super().__init__(3, 1)
//         self.crossover_rate = crossover_rate
//         self.F = F
//         self._rng = np.random.default_rng(seed)

//     def _perform_crossover(self, parents: list[list[int | float]]) -> list[int | float]:
//         child = np.copy(parents[0])
//         jr = self._rng.integers(len(child))
//         for i in range(len(child)):
//             if self._rng.random() < self.crossover_rate or i == jr:
//                 child[i] += self.F * (parents[1][i] - parents[2][i])
//         return child
