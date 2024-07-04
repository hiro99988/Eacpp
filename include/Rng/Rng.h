#ifndef Rng_h
#define Rng_h

#include <cstdint>
#include <eigen3/Eigen/Core>
#include <random>
#include <tuple>
#include <vector>

namespace Eacpp {

class Rng {
   public:
    Rng() : mt(std::random_device()()){};
    Rng(std::uint_fast32_t seed) : mt(seed){};

    int Integer(const int max) const;
    int Integer(int min, int max) const;
    std::vector<int> Integers(int min, int max, const int size, bool replace = true) const;

    double Uniform(double min, double max) const;
    Eigen::ArrayXd Uniform(double min, double max, const int size) const;
    Eigen::ArrayXXd Uniform(double min, double max, const std::tuple<int, int> size) const;

    double Random() const;
    Eigen::ArrayXd Random(const int size) const;
    Eigen::ArrayXXd Random(const std::tuple<int, int> size) const;

    template <typename T>
    Eigen::ArrayXX<T> Choice(Eigen::ArrayXX<T>, int size, bool replace = true) const;

   private:
    std::mt19937 mt;
};

}  // namespace Eacpp

#endif