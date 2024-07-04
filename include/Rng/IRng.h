#ifndef IRNG_H
#define IRNG_H

#include <cstdint>
#include <eigen3/Eigen/Core>
#include <random>
#include <tuple>
#include <vector>

namespace Eacpp {

struct IRng {
    virtual int Integer(const int max) const = 0;
    virtual int Integer(int min, int max) const = 0;
    virtual std::vector<int> Integers(int min, int max, const int size, bool replace = true) const = 0;

    virtual double Uniform(double min, double max) const = 0;
    virtual Eigen::ArrayXd Uniform(double min, double max, const int size) const = 0;
    virtual Eigen::ArrayXXd Uniform(double min, double max, const std::tuple<int, int> size) const = 0;

    virtual double Random() const = 0;
    virtual Eigen::ArrayXd Random(const int size) const = 0;
    virtual Eigen::ArrayXXd Random(const std::tuple<int, int> size) const = 0;

    // template <typename T>
    // virtual Eigen::ArrayXX<T> Choice(Eigen::ArrayXX<T>, int size, bool replace = true) const;
};

}  // namespace Eacpp

#endif