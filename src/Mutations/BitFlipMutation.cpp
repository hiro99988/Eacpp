#include "Mutations/BitFlipMutation.h"

#include "Utils/Rng.h"

namespace Eacpp {

void BitFlipMutation::Mutate(Eigen::ArrayXi& individual) const {
    for (int i = 0; i < individual.size(); i++) {
        if (_rng.Random() < mutationRate) {
            individual(i) = 1 - individual(i);
        }
    }
}

}  // namespace Eacpp