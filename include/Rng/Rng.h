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
    Eigen::ArrayXXd Uniform(double min, double max, const std::tuple<int, int> size) override;

    double Random() override;
    Eigen::ArrayXd Random(const int size) override;
    Eigen::ArrayXXd Random(const std::tuple<int, int> size) override;

    Eigen::ArrayXXi Choice(Eigen::ArrayXXi array, int size, bool replace = true) override {
        return Choice(array, size, replace);
    }
    Eigen::ArrayXXd Choice(Eigen::ArrayXXd array, int size, bool replace = true) override {
        return Choice(array, size, replace);
    }

   private:
    std::mt19937 _mt;

    template <typename T>
    Eigen::ArrayXX<T> Choice(Eigen::ArrayXX<T> array, int size, bool replace = true) {
        Eigen::ArrayXX<T> result(array.rows(), size);
        auto v = Integers(0, array.cols() - 1, size, replace);
        return result;
    };
};

}  // namespace Eacpp

#endif