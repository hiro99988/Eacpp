#ifndef BitFlipMutation_h
#define BitFlipMutation_h

#include "Mutations/IMutation.h"
#include "Rng/Rng.h"

namespace Eacpp {
class BitFlipMutation : public IMutation<int> {
   public:
    double mutationRate;

    explicit BitFlipMutation(double mutationRate) : mutationRate(mutationRate), _rng() {}
    BitFlipMutation(double mutationRate, std::uint_fast32_t seed) : mutationRate(mutationRate), _rng(seed) {}

    void Mutate(Eigen::ArrayXi& individual) const override;

   private:
    Rng _rng;
};

}  // namespace Eacpp

#endif