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
    Rng() : mt(std::random_device()()){};
    Rng(std::uint_fast32_t seed) : mt(seed){};

    int Integer(const int max) const override;
    int Integer(int min, int max) const override;
    std::vector<int> Integers(int min, int max, const int size, bool replace) const override;

    double Uniform(double min, double max) const override;
    Eigen::ArrayXd Uniform(double min, double max, const int size) const override;
    Eigen::ArrayXXd Uniform(double min, double max, const std::tuple<int, int> size) const override;

    double Random() const override;
    Eigen::ArrayXd Random(const int size) const override;
    Eigen::ArrayXXd Random(const std::tuple<int, int> size) const override;

    template <typename T>
    Eigen::ArrayXX<T> Choice(Eigen::ArrayXX<T>, int size, bool replace) const;

   private:
    std::mt19937 mt;
};

}  // namespace Eacpp

#endif