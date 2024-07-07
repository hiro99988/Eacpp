#ifndef BitFlipMutation_h
#define BitFlipMutation_h

#include <eigen3/Eigen/Core>
#include <iostream>

#include "Mutations/IMutation.h"
#include "Rng/IRng.h"
#include "Rng/Rng.h"

namespace Eacpp {
class BitFlipMutation : public IMutation<int> {
   public:
    double mutationRate;

    explicit BitFlipMutation(double mutationRate) : mutationRate(mutationRate) {
        _rng = new Rng();
        _isRngCreated = true;
    }
    BitFlipMutation(double mutationRate, IRng* rng) : mutationRate(mutationRate), _rng(rng) {}
    ~BitFlipMutation() {
        if (_isRngCreated) {
            delete _rng;
        }
    }

    void Mutate(Eigen::ArrayXi& individual) const override;

   private:
    IRng* _rng;
    bool _isRngCreated = false;
};

}  // namespace Eacpp

#endif