#ifndef Rng_h
#define Rng_h

#include <cstdint>
#include <eigen3/Eigen/Core>
#include <random>
#include <tuple>
#include <vector>

#include "Rng/IRng.h"

namespace Eacpp {

class Rng : public IRng {
   public:
    Rng() : _mt(std::random_device()()) {}
    Rng(std::uint_fast32_t seed) : _mt(seed) {}

    int Integer(const int max) override;
    int Integer(int min, int max) override;
    std::vector<int> Integers(int min, int max, const int size, bool replace = true) override;

    double Uniform(double min, double max) override;
    Eigen::ArrayXd Uniform(double min, double max, const int size) override;
    std::vector<Eigen::ArrayXd> Uniform(double min, double max, const std::pair<int, int> size) override;

    double Random() override;
    Eigen::ArrayXd Random(const int size) override;
    std::vector<Eigen::ArrayXd> Random(const std::pair<int, int> size) override;

    std::vector<int> Choice(const std::vector<int>& array, const int size, const bool replace = true) override;

   private:
    std::mt19937 _mt;
};

}  // namespace Eacpp

#endif