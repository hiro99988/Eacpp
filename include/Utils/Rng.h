#ifndef Rng_h
#define Rng_h

#include <cstdint>
#include <random>
#include <vector>

namespace Eacpp {

class Rng {
   public:
    Rng() : mt(std::random_device()()){};
    Rng(std::uint_fast32_t seed) : mt(seed){};

    int Integer(const int max);
    int Integer(int min, int max);
    std::vector<int> Integers(int min, int max, const int size, bool replace);

   private:
    std::mt19937 mt;
};

}  // namespace Eacpp

#endif