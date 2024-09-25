#include "Mutations/BitFlipMutation.h"

#include "Individual/Individual.h"
#include "Rng/Rng.h"

namespace Eacpp {
void BitFlipMutation::Mutate(Individuali& individual) const {
    for (int i = 0; i < individual.solution.size(); i++) {
        if (_rng->Random() < mutationRate) {
            individual.solution(i) = 1 - individual.solution(i);
        }
    }
}

}  // namespace Eacpp