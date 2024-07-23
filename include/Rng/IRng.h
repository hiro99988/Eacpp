#ifndef IRNG_H
#define IRNG_H

#include <cstdint>
#include <eigen3/Eigen/Core>
#include <random>
#include <tuple>
#include <vector>

namespace Eacpp {

struct IRng {
    virtual ~IRng() {}

    virtual int Integer(const int max) = 0;
    virtual int Integer(int min, int max) = 0;
    virtual std::vector<int> Integers(int min, int max, const int size, bool replace) = 0;

    virtual double Uniform(double min, double max) = 0;
    virtual Eigen::ArrayXd Uniform(double min, double max, const int size) = 0;
    virtual Eigen::ArrayXXd Uniform(double min, double max, const std::tuple<int, int> size) = 0;

    virtual double Random() = 0;
    virtual Eigen::ArrayXd Random(const int size) = 0;
    virtual Eigen::ArrayXXd Random(const std::tuple<int, int> size) = 0;

    virtual Eigen::ArrayXi Choice(const Eigen::ArrayXi& array, const int size, const bool replace) = 0;
    virtual Eigen::ArrayXd Choice(const Eigen::ArrayXd& array, const int size, const bool replace) = 0;
};

}  // namespace Eacpp

#endif