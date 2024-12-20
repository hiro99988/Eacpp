#pragma once

#include <memory>

#include "Individual.h"
#include "Repairs/IRepair.h"
#include "Samplings/ISampling.h"

namespace Eacpp {

template <typename T>
class SamplingRepair : public IRepair<T> {
   public:
    SamplingRepair(std::shared_ptr<ISampling<T>> sampling)
        : sampling(sampling) {}

    void Repair(Individual<T>& individual) override {
        individual = sampling->Sample(1, individual.solution.size())[0];
    }

   private:
    std::shared_ptr<ISampling<T>> sampling;
};

}  // namespace Eacpp
