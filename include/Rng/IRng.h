#ifndef IRNG_H
#define IRNG_H

#include <cstdint>
#include <eigen3/Eigen/Core>
#include <random>
#include <tuple>
#include <vector>

namespace Eacpp {

struct IRng {
    virtual int Integer(const int max) const;
    virtual int Integer(int min, int max) const;
    virtual std::vector<int> Integers(int min, int max, const int size, bool replace = true) const;

    virtual double Uniform(double min, double max) const;
    virtual Eigen::ArrayXd Uniform(double min, double max, const int size) const;
    virtual Eigen::ArrayXXd Uniform(double min, double max, const std::tuple<int, int> size) const;

    virtual double Random() const;
    virtual Eigen::ArrayXd Random(const int size) const;
    virtual Eigen::ArrayXXd Random(const std::tuple<int, int> size) const;

    // template <typename T>
    // virtual Eigen::ArrayXX<T> Choice(Eigen::ArrayXX<T>, int size, bool replace = true) const;
};

}  // namespace Eacpp

#endif