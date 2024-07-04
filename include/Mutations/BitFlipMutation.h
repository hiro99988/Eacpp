#ifndef BitFlipMutation_h
#define BitFlipMutation_h

#include "Mutations/IMutation.h"
#include "Rng/IRng.h"
#include "Rng/Rng.h"

namespace Eacpp {
class BitFlipMutation : public IMutation<int> {
   public:
    double mutationRate;

    explicit BitFlipMutation(double mutationRate) : mutationRate(mutationRate), _rng(Rng()) {}
    BitFlipMutation(double mutationRate, IRng rng) : mutationRate(mutationRate), _rng(rng) {}

    void Mutate(Eigen::ArrayXi& individual) const override;

   private:
    IRng& _rng;
};

}  // namespace Eacpp

#endif